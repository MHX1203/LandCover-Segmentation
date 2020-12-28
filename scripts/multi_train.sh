#!/bin/bash

echo "Run Training"
/home/mhxin/miniconda3/envs/torch/bin/python /home/mhxin/桌面/SegProject/train.py --config_file config/nested_unet_config.yaml
echo "Finish nested_unet"
/home/mhxin/miniconda3/envs/torch/bin/python /home/mhxin/桌面/SegProject/train.py --config_file config/attention_unet_config.yaml
echo "Finish attention_unet"
/home/mhxin/miniconda3/envs/torch/bin/python /home/mhxin/桌面/SegProject/train.py --config_file config/r2u_config.yaml
echo "Finish r2u"
/home/mhxin/miniconda3/envs/torch/bin/python /home/mhxin/桌面/SegProject/train.py --config_file config/rec_attention_unet_config.yaml
echo "Finish rec_attention_unet"