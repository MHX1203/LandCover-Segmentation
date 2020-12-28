#!/bin/bash

echo "Run inference"
/home/mhxin/miniconda3/envs/torch/bin/python /home/mhxin/桌面/SegProject/predict.py --data_dir ../img_testA --checkpoint checkpoints/unet-4_MIOU-0.8950199535228898_CareMIOU-0.8936372012363512.pth --save_dir /home/mhxin/桌面/ccf_baidu_remote_sense --batch_size 128 --use_gpu true

echo "Done"