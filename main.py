import os, sys, pdb
import argparse
from models import get_model
from data import make_data_loader
import warnings
from trainer import Trainer
import torch
import torch.backends.cudnn as cudnn
import random


parser = argparse.ArgumentParser(description='PyTorch Training for Multi-label Image Classification')

''' Fixed in general '''
parser.add_argument('--data_root_dir', default='./datasets/', type=str, help='save path')
parser.add_argument('--image-size', '-i', default=448, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch_step', default=[30, 40], type=int, nargs='+', help='number of epochs to change learning rate')
# parser.add_argument('--device_ids', default=[0], type=int, nargs='+', help='number of epochs to change learning rate')
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='INT', help='number of data loading workers (default: 4)')
parser.add_argument('--display_interval', default=200, type=int, metavar='M', help='display_interval')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float)
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float, metavar='LRP', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max_clip_grad_norm', default=10.0, type=float, metavar='M', help='max_clip_grad_norm')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')

''' Train setting '''
parser.add_argument('--data', metavar='NAME', help='dataset name (e.g. COCO2014')
parser.add_argument('--model_name', type=str, default='ADD_GCN')
parser.add_argument('--save_dir', default='./checkpoint/COCO2014/', type=str, help='save path')

''' Val or Tese setting '''
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


def main(args):

    if args.seed is not None:
        print ('* absolute seed: {}'.format(args.seed))
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    is_train = True if not args.evaluate else False
    train_loader, val_loader, num_classes = make_data_loader(args, is_train=is_train)

    model = get_model(num_classes, args)

    criterion = torch.nn.MultiLabelSoftMarginLoss()

    trainer = Trainer(model, criterion, train_loader, val_loader, args)
    
    if is_train:
        trainer.train()
    else:
        trainer.validate()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
