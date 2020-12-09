import os, sys, pdb
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import torchnet as tnt
# import torchvision.transforms as transforms
from torch.autograd import Variable
# from torch.optim import lr_scheduler
from util import AverageMeter, AveragePrecisionMeter
from datetime import datetime
# from pprint import pprint
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, criterion, train_loader, val_loader, args):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        # pprint (self.args)
        print('--------Args Items----------')
        for k, v in vars(self.args).items():
            print('{}: {}'.format(k, v))
        print('--------Args Items----------\n')

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.SGD(self.model.get_config_optim(self.args.lr, self.args.lrp), 
                                        lr=self.args.lr, 
                                        momentum=self.args.momentum, 
                                        weight_decay=self.args.weight_decay)
        # self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, self.args.epoch_step, gamma=0.1)

    def initialize_meters(self):
        self.meters = {}
        # meters
        self.meters['loss'] = AverageMeter('loss')
        self.meters['ap_meter'] = AveragePrecisionMeter()
        # time measure
        self.meters['batch_time'] = AverageMeter('batch_time')
        self.meters['data_time'] = AverageMeter('data_time')

    def initialization(self, is_train=False):
        """ initialize self.model and self.criterion here """
        
        if is_train:
            self.start_epoch = 0
            self.epoch = 0
            self.end_epoch = self.args.epochs
            self.best_score = 0.
            self.lr_now = self.args.lr

            # initialize some settings
            self.initialize_optimizer_and_scheduler()

        self.initialize_meters()

        # load checkpoint if args.resume is a valid checkpint file.
        if os.path.isfile(self.args.resume) and self.args.resume.endswith('pth'):
            self.load_checkpoint()
        
        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.criterion = self.criterion.cuda()
            #     self.train_loader.pin_memory = True
            # self.val_loader.pin_memory = True

    def reset_meters(self):
        for k, v in self.meters.items():
            self.meters[k].reset()

    def on_start_epoch(self):
        self.reset_meters()

    def on_end_epoch(self, is_train=False):

        if is_train:
            # maybe you can do something like 'print the training results' here.
            return 
        else:
            # map = self.meters['ap_meter'].value().mean()
            ap =  self.meters['ap_meter'].value()
            print (ap)
            map = ap.mean()
            loss = self.meters['loss'].average()
            data_time = self.meters['data_time'].average()
            batch_time = self.meters['batch_time'].average()

            OP, OR, OF1, CP, CR, CF1 = self.meters['ap_meter'].overall()
            OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.meters['ap_meter'].overall_topk(3)

            print('* Test\nLoss: {loss:.4f}\t mAP: {map:.4f}\t' 
                    'Data_time: {data_time:.4f}\t Batch_time: {batch_time:.4f}'.format(
                    loss=loss, map=map, data_time=data_time, batch_time=batch_time))
            print('OP: {OP:.3f}\t OR: {OR:.3f}\t OF1: {OF1:.3f}\t'
                    'CP: {CP:.3f}\t CR: {CR:.3f}\t CF1: {CF1:.3f}'.format(
                    OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            print('OP_3: {OP:.3f}\t OR_3: {OR:.3f}\t OF1_3: {OF1:.3f}\t'
                    'CP_3: {CP:.3f}\t CR_3: {CR:.3f}\t CF1_3: {CF1:.3f}'.format(
                    OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))
                    
            return map

    def on_forward(self, inputs, targets, is_train):
        inputs = Variable(inputs).float()
        targets = Variable(targets).float()

        if not is_train:
            with torch.no_grad():
                outputs1, outputs2 = self.model(inputs)
        else:
            outputs1, outputs2 = self.model(inputs)
        outputs = (outputs1 + outputs2) / 2

        loss = self.criterion(outputs, targets)
        self.meters['loss'].update(loss.item(), inputs.size(0))

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_clip_grad_norm)
            self.optimizer.step()

        return outputs
    
    def adjust_learning_rate(self):
        """ Sets learning rate if it is needed """
        lr_list = []
        decay = 0.1 if sum(self.epoch == np.array(self.args.epoch_step)) > 0 else 1.0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])

        return np.unique(lr_list)

    def train(self):
        self.initialization(is_train=True)

        for epoch in range(self.start_epoch, self.end_epoch):
            self.lr_now = self.adjust_learning_rate()
            print ('Lr: {}'.format(self.lr_now))

            self.epoch = epoch
            # train for one epoch
            self.run_iteration(self.train_loader, is_train=True)

            # evaluate on validation set
            score = self.run_iteration(self.val_loader, is_train=False)

            # record best score, save checkpoint and result
            is_best = score > self.best_score
            self.best_score = max(score, self.best_score)
            checkpoint = {
                'epoch': epoch + 1, 
                'model_name': self.args.model_name,
                'state_dict': self.model.module.state_dict() if torch.cuda.is_available() else self.model.state_dict(),
                'best_score': self.best_score
                }
            model_dir = self.args.save_dir
            # assert os.path.exists(model_dir) == True
            self.save_checkpoint(checkpoint, model_dir, is_best)
            self.save_result(model_dir, is_best)

            print(' * best mAP={best:.4f}'.format(best=self.best_score))

        return self.best_score

    def run_iteration(self, data_loader, is_train=True):
        self.on_start_epoch()

        if not is_train:
            data_loader = tqdm(data_loader, desc='Validate')
            self.model.eval()
        else:
            self.model.train()

        st_time = time.time()
        for i, data in enumerate(data_loader):

            # measure data loading time
            data_time = time.time() - st_time
            self.meters['data_time'].update(data_time)

            # inputs, targets, targets_gt, filenames = self.on_start_batch(data)
            inputs = data['image']
            targets = data['target']

            # for voc
            labels = targets.clone()
            targets[targets==0] = 1
            targets[targets==-1] = 0

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = self.on_forward(inputs, targets, is_train=is_train)

            # measure elapsed time
            batch_time = time.time() - st_time
            self.meters['batch_time'].update(batch_time)

            self.meters['ap_meter'].add(outputs.data, labels.data, data['name'])
            st_time = time.time()

            if is_train and i % self.args.display_interval == 0:
                print ('{}, {} Epoch, {} Iter, Loss: {:.4f}, Data time: {:.4f}, Batch time: {:.4f}'.format(
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  self.epoch+1, i, 
                        self.meters['loss'].value(), self.meters['data_time'].value(), 
                        self.meters['batch_time'].value()))

        return self.on_end_epoch(is_train=is_train)

    def validate(self):
        self.initialization(is_train=False)

        map = self.run_iteration(self.val_loader, is_train=False)

        model_dir = os.path.dirname(self.args.resume)
        assert os.path.exists(model_dir) == True
        self.save_result(model_dir, is_best=False)

        return map

    def load_checkpoint(self):
        print("* Loading checkpoint '{}'".format(self.args.resume))
        checkpoint = torch.load(self.args.resume)
        self.start_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        model_dict = self.model.state_dict()
        for k, v in checkpoint['state_dict'].items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
            else:
                print ('\tMismatched layers: {}'.format(k))
        self.model.load_state_dict(model_dict)

    def save_checkpoint(self, checkpoint, model_dir, is_best=False):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # filename = 'Epoch-{}.pth'.format(self.epoch)
        filename = 'checkpoint.pth'
        res_path = os.path.join(model_dir, filename)
        print('Save checkpoint to {}'.format(res_path))
        torch.save(checkpoint, res_path)
        if is_best:
            filename_best = 'checkpoint_best.pth'
            res_path_best = os.path.join(model_dir, filename_best)
            shutil.copyfile(res_path, res_path_best)

    def save_result(self, model_dir, is_best=False):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # filename = 'results.csv' if not is_best else 'best_results.csv'
        filename = 'results.csv'
        res_path = os.path.join(model_dir, filename)
        print('Save results to {}'.format(res_path))
        with open(res_path, 'w') as fid:
            for i in range(self.meters['ap_meter'].scores.shape[0]):
                fid.write('{},{},{}\n'.format(self.meters['ap_meter'].filenames[i], 
                    ','.join(map(str,self.meters['ap_meter'].scores[i].numpy())), 
                    ','.join(map(str,self.meters['ap_meter'].targets[i].numpy()))))
        
        if is_best:
            filename_best = 'output_best.csv'
            res_path_best = os.path.join(model_dir, filename_best)
            shutil.copyfile(res_path, res_path_best)
