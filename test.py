from data.dataset import SatImageDataset
from torch.utils.data import DataLoader
from data.utils import preprocess, get_augmentations
from glob import glob
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch as t
import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from utils.visualizer import vis_mask
import yaml
from tqdm import tqdm
from utils.loss import FocalLoss
content = yaml.load(open('config/default.yaml', 'r').read(), Loader=yaml.SafeLoader)
# ---------------------------------------------------------------- #
# 数据集
# image_dir = '../train_data/img_train'
# label_dir = '../train_data/lab_train'

# img_list = glob(image_dir + '/*')

# img_list = sorted(img_list)
# lab_list = glob(label_dir + '/*')

# lab_list = sorted(lab_list)


# dataset = SatImageDataset(img_list, preprocess=preprocess)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# tbar = tqdm(enumerate(dataloader), desc='infer')
# for i, (img_path, img) in tbar:
#     if i > 5:
#         break
#     print(len(img_path), img_path[1], img.shape)
# writer = SummaryWriter('./temp_logs')
# print(content['data']['color_dicts'])
# for i, (image, label) in enumerate(dataloader):
#     if i > 5:
#         break

    
#     label = vis_mask(label, content['data']['color_dicts'])
#     label = make_grid(label, nrow=2, padding=10)
#     writer.add_image('img', label, i)


criterion = FocalLoss().cuda()
a = t.rand(4, 3, 256, 256).cuda()
b = t.randint(0, 3, (4, 256, 256)).type(t.long).cuda()
loss = criterion(a, b)
print(loss)