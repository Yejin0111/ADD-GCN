import os, sys, pdb
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from coco import COCO2014
from voc import VOC2007, VOC2012

data = sys.argv[1]


def collate_fn(batch):
    ret_batch = dict()
    for k in batch[0].keys():
        if k == 'image' or k == 'target':
            ret_batch[k] = torch.cat([b[k].unsqueeze(0) for b in batch])
        else:
            ret_batch[k] = [b[k] for b in batch]
    return ret_batch


transform = transforms.Compose([
        transforms.RandomResizedCrop(448, scale=(0.1, 1.5), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# COCO2014
# train_dataset = COCO2014(data, phase='train', transform=transform)
# val_dataset = COCO2014(data, phase='val', transform=transform)

# VOC2007
# train_dataset = VOC2007(data, phase='trainval', transform=transform)
# val_dataset = VOC2007(data, phase='test', transform=transform)

# VOC2012
train_dataset = VOC2012(data, phase='trainval', transform=transform)
val_dataset = VOC2012(data, phase='test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, 
                            pin_memory=True, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, 
                            pin_memory=True, collate_fn=collate_fn)

for data in train_loader:
    pdb.set_trace()



pdb.set_trace()
