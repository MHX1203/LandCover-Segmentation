import torch as t
import torch.nn as nn
from model.builder import build_model
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
from data.builder import build_loader
import os
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from glob import glob
from data.utils import Augment, preprocess, get_augmentations
from datetime import datetime
from tqdm import tqdm
from torchvision.utils import make_grid
import json
from utils.visualizer import vis_mask
import random
from utils.loss import FocalLoss
def setup_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(20)

class Trainer:
    def __init__(self, **kwargs):
        
        if kwargs['use_gpu']:
            print('Using GPU')
            self.device = t.device('cuda')
        else:
            self.device = t.device('cuda')

        # data
        if not os.path.exists(kwargs['root_dir'] + '/train_image_list'):
            image_list = glob(kwargs['root_dir'] + '/img_train/*')
            image_list = sorted(image_list)

            mask_list = glob(kwargs['root_dir'] + '/lab_train/*')
            mask_list = sorted(mask_list)

        else:
            image_list = open(kwargs['root_dir'] + '/train_image_list', 'r').readlines()
            image_list = [line.strip() for line in image_list]
            image_list = sorted(image_list)
            mask_list = open(kwargs['root_dir'] + '/train_label_list', 'r').readlines()
            mask_list = [line.strip() for line in mask_list]
            mask_list = sorted(mask_list)

        print(image_list[-5:], mask_list[-5:])

        if kwargs['augs']:
            augs = Augment(get_augmentations())
        else:
            augs = None

        self.train_loader, self.val_loader = build_loader(image_list, mask_list, kwargs['test_size'], augs, preprocess, \
            kwargs['num_workers'], kwargs['batch_size'])

        self.model = build_model(kwargs['in_channels'], kwargs['num_classes'], kwargs['model_name']).to(self.device)

        if kwargs['resume']:
            try:
                self.model.load_state_dict(t.load(kwargs['resume']))
                
            except Exception as e:
                self.model.load_state_dict(t.load(kwargs['resume'])['model'])

            print(f'load model from {kwargs["resume"]} successfully')
        if kwargs['loss'] == 'CE':
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif kwargs['loss'] == 'FL':
            self.criterion = FocalLoss().to(self.device)
        else:
            raise NotImplementedError
        if kwargs['use_sgd']:
            self.optimizer = SGD(self.model.parameters(), lr=kwargs['lr'], momentum=kwargs['momentum'], nesterov=True, weight_decay=float(kwargs['weight_decay'])) 

        else:
            self.optimizer = Adam(self.model.parameters(), lr=kwargs['lr'], weight_decay=float(kwargs['weight_decay']))

        self.lr_planner = CosineAnnealingWarmRestarts(self.optimizer, 100, T_mult=2, eta_min=1e-6, verbose=True)

        log_dir = os.path.join(kwargs['log_dir'], datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        self.writer = SummaryWriter(log_dir, comment=f"LR-{kwargs['lr']}_BatchSize-{kwargs['batch_size']}_ModelName-{kwargs['model_name']}")

        self.name = kwargs['model_name']
        self.epoch = kwargs['epoch']

        self.start_epoch = kwargs['start_epoch']

        self.val_interval = kwargs['val_interval']

        self.log_interval = kwargs['log_interval']

        self.num_classes = kwargs['num_classes']

        self.checkpoints = kwargs['checkpoints']

        self.color_dicts = kwargs['color_dicts']


        # , format='%Y-%m-%d %H:%M:%S',
        logging.basicConfig(filename=log_dir + '/log.log', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

        s = '\n\t\t'
        for k, v in kwargs.items():
            s += f"{k}: \t{v}\n\t\t"

        logging.info(s)

    def train(self):
        max_miou = 0
        for epoch in range(self.start_epoch, self.epoch):
            logging.info(f'Train Epoch-{epoch}',)
            self.model.train()
            history = self.step(epoch)

            logging.info(f'Loss: {history[0]}; MIOU: {history[1]}, CareMIOU: {history[1]}')

            # write to tensorboard, log it
            logging.info(f'Val Epoch - {epoch}')
            history = self.eval()
            logging.info(f'MIOU: {history[0]}; CareMIOU: {history[1]}')
            self.writer.add_scalar('MIOU/Val', history[0], epoch)

            # 保存最好模型
            if history[0] > max_miou:
                self.save_model(f'{self.name}-{epoch}_MIOU-{history[0]}_CareMIOU-{history[1]}.pth')
                max_miou = history[0]
            

    def step(self, epoch):

        total_loss = 0
        correct = 0
        count = 0
        care_total = 0
        care_correct = 0
        tbar = tqdm(enumerate(self.train_loader), desc=f'Epoch:{epoch}', unit='batch', total=len(self.train_loader))
        for it, (image, mask) in tbar:
            image = image.to(self.device)
            mask = mask.to(self.device)
            count += mask.numel()
            care_mask = mask < 7
            care_total += care_mask.sum().item()

            self.optimizer.zero_grad()

            pred = self.model(image)

            correct += (pred.argmax(1) == mask).sum().item()
            care_correct += (pred.argmax(1)[care_mask] == mask[care_mask]).sum().item()

            loss = self.criterion(pred, mask)

            total_loss += loss.item()

            loss.backward()

            self.optimizer.step()

            tbar.set_description(f'[Iter/Total: {it}/{len(self.train_loader)}, Loss: {round(total_loss / (it + 1), 4)}, Accuracy/IOU: {round(float(correct) / count, 4)}, Care-IOU: {round(float(care_correct) / care_total, 4)}]')
            global_step = (epoch - self.start_epoch) * len(self.train_loader) + it
            if global_step % self.log_interval == 0:
                
                self.writer.add_scalar('Loss/train', total_loss / (it + 1), global_step)
                self.writer.add_scalar('Lr/Train', self.optimizer.param_groups[0]['lr'], global_step)
                self.writer.add_scalar('MIOU/Train', float(correct) / count, global_step)
                self.writer.add_scalar('Care-MIOU/Train', float(care_correct) / care_total, global_step)
                self.lr_planner.step()

            # if (it + 1) % (self.log_interval * 10) == 0:
            if global_step % (self.log_interval * 10) == 0:
                global_step = (epoch - self.start_epoch) * len(self.train_loader) + it
                if image.shape[0] < 4:
                    mask = vis_mask(mask.cpu(), self.color_dicts)
                    self.writer.add_image('Masks/True', make_grid(mask, nrow=2, padding=10), global_step)
                
                    pred = vis_mask(pred.argmax(1).cpu(), self.color_dicts)
                    self.writer.add_image('Masks/Pred', make_grid(pred, nrow=2, padding=10), global_step)  
                else:
                    mask = mask.cpu()
                    idx = t.randint(0, mask.shape[0], (4, ))
                    mask = vis_mask(mask.index_select(0, idx), self.color_dicts)
                    self.writer.add_image('Masks/True', make_grid(mask, nrow=2, padding=10), global_step)

                    pred = vis_mask(pred.argmax(1).cpu().index_select(0, idx), self.color_dicts)
                    self.writer.add_image('Masks/Pred', make_grid(pred, nrow=2, padding=10), global_step)  

        return total_loss / len(self.train_loader), float(correct) / count, float(care_correct) / care_total


    def save_model(self, save_name, save_opt=True):
        dicts = {}

        dicts['model'] = self.model.state_dict()

        dicts['optimizer'] = self.optimizer.state_dict()

        os.makedirs(self.checkpoints, exist_ok=True)

        save_path = os.path.join(self.checkpoints, save_name)
        t.save(dicts, save_path)

        logging.info("Save model to {}".format(save_path))


    def load_model(self, model_path, load_opt=True):
        if not os.path.exists(model_path):
            print(f'{model_path} does not exist, please check it')
        try:
            dicts = t.load(model_path)
        except Exception as e:
            print(e.args)
            print('Load model failed')
            return

        self.optimizer.load_state_dict(dicts['optimizer'])
        self.model.load_state_dict(dicts['model'])
        logging.info(f'Load model from {model_path} successfully...')


    def eval(self):
        self.model.eval()
        tbar = tqdm(enumerate(self.val_loader), desc='Model Evaluation', unit='batch', total=len(self.val_loader))
        correct = 0
        total = 0
        care_total = 0
        care_correct = 0
        self.model.eval()
        with t.no_grad():
            for it, (image, mask) in tbar:
                image = image.to(self.device)
                # mask = mask.to(self.device)

                pred = self.model(image).cpu()

                total += mask.numel()
                
                care_mask = mask < 7
                care_total += care_mask.sum().item()


                correct += (mask == pred.argmax(1)).sum().item()
                # print(mask[care_mask].shape, pred.argmax(1)[care_mask].shape)
                true_mask = (mask[care_mask] == pred.argmax(1)[care_mask])
                care_correct += true_mask.sum().item()

        return float(correct) / total, float(care_correct) / care_total
