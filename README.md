# LandCover-Segmentation

this project from [2020 CCF BDCI 遥感影像地块分割](https://aistudio.baidu.com/aistudio/competition/detail/54)

# Usage

for training
> python train.py --config_file ./config/default.yaml

for inference
> python predict.py --data_dir ../img_testA --checkpoint checkpoints/best.pth --save_dir .  --batch_size 128 --use_gpu true
