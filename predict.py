import torch as t
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import argparse
from glob import glob
from data.dataset import SatImageDataset
from data.utils import preprocess
from model.builder import build_model
from time import time 

def str2bool(s):
    if 't' in s.lower() or 'true' in s.lower():
        return True
    return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='test data directory')

    parser.add_argument('--save_dir', type=str, help='inference result saved directory')

    parser.add_argument('--use_gpu', type=str2bool, help='whether use gpu', default='true')

    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint path')

    parser.add_argument('--batch_size', type=int, required=True, help='batch size, default: 512')

    return parser.parse_args()

def save_preds(preds, image_paths, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    preds = preds.detach().cpu().numpy()
    preds[preds == 7] = 0
    for i, pred in enumerate(preds):
        name = os.path.split(image_paths[i])[-1][:-4]
        Image.fromarray(pred.astype(np.uint8)).save(save_dir + f'/{name}.png')


def read_image(img_path, target_size=256):
    image = Image.open(img_path)

    image = image.resize((target_size, target_size), Image.BILINEAR)
    image = np.array(image, dtype=np.uint8)
    image = image / 255.
    image = image.transpose(2, 0, 1)

    mean= [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    image = (image - np.array(mean, dtype=np.float32)[:, None, None]) / np.array(std, dtype=np.float32)[:, None, None]

    return t.from_numpy(image).unsqueeze(0).type(t.float32)

def main():
    args = parse_args()
    if not os.path.exists(args.data_dir):
        print('data_dir is not exists, please check it\nexit...')
        exit(0)

    if args.use_gpu:
        device = t.device('cuda')
        print('use GPU')
    else:
        device = t.device('cpu')

    
    model = build_model(3, 8, model_name='unet').to(device)

    model.load_state_dict(t.load(args.checkpoint)['model'])

    model.eval()


    if args.data_dir.lower().endswith('png') or args.data_dir.lower().endswith('jpg'):
        img = read_image(args.data_dir)
        with t.no_grad():
            pred = model(img.to(device))
        save_preds(pred.argmax(1), [args.data_dir], args.save_dir)
        # print('Done')
        exit(0)
    if os.path.isdir(args.data_dir):
        image_list = glob(args.data_dir + '/*.jpg')
    else:
        image_list = open(args.data_dir + '/*.jpg').read().split('\n')
    
    test_set = SatImageDataset(image_list, shuffle=False, preprocess=preprocess)

    test_loader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.batch_size//10, drop_last=False)

    
    with t.no_grad():
        tbar = tqdm(enumerate(test_loader), desc='infer')
        for i, (img_path, img) in tbar:
            t1 = time()
            img = img.to(device)
            preds = model(img).argmax(1)
            save_preds(preds, img_path, args.save_dir) 
    print('Done')

if __name__ == "__main__":
    main()
