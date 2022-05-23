import argparse
import collections
import torch
import random
import numpy as np
import TireProject_official.detectWear.utils.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from TireProject_official.detectWear.parse_config import ConfigParser
from trainer.trainer import Trainer
from utils import prepare_device
import torch.nn as nn


# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def main(config):

    # setup data_loader instances
    train_data_loader = config.init_obj('train_data_loader', module_data)
    val_data_loader = config.init_obj('val_data_loader', module_data)

    # attention_loader instances
    attention_data_loader = config.init_obj('attention_data_loader', module_data)
    attention_val_data_loader = config.init_obj('attention_val_data_loader', module_data)

    # build model
    model = module_arch.resnet101()
    checkpoint = torch.load('model/resnet101-5d3b4d8f.pth')
    model.load_state_dict(checkpoint)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs+1, 3)  # +1 is concatenated attention mask layer

    print(model)


    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_data_loader,
                      valid_data_loader=val_data_loader,
                      attention_data_loader=attention_data_loader,
                      attention_val_data_loader= attention_val_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
