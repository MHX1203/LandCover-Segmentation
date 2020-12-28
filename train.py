import yaml
import os
import argparse
from trainer import Trainer
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='./config/default.yaml')

    return parser.parse_args()


def read_yaml(yaml_file):
    content = yaml.load(open(yaml_file, 'r').read(), Loader=yaml.SafeLoader)

    return {**content['data'], **content['train'], **content['model']}

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.config_file):
        print(f'{args.config} not exists')
        exit(0)

    params = read_yaml(args.config_file)

    trainer = Trainer(**params)

    trainer.train()


