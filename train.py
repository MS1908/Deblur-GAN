import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from functools import partial
from dataset_utils.dataset_loader import load_dataset_from_config
from train_utils.trainer_utils import Trainer

torch.manual_seed(42)
np.random.seed(42)


def train(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    batch_size = config.pop('batch_size')
    get_dataloader = partial(DataLoader,
                             batch_size=batch_size,
                             num_workers=0,
                             shuffle=True, drop_last=True)
    datasets = map(config.pop, ('train', 'val'))
    datasets = map(load_dataset_from_config, datasets)
    train, val = map(get_dataloader, datasets)
    trainer = Trainer(config, train=train, val=val)
    trainer.train()


if __name__ == '__main__':
    train()
