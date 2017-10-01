import sys
import argparse
import os.path
import importlib
from random import shuffle
from PIL import Image

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.transforms as t
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import utils
import metric

##########################
# Task-specific options. #
##########################
parser = argparse.ArgumentParser(description='Large-scale image classification')
parser.add_argument('--db', default='imagenet', metavar='NAME', type=str,
                    help='dataset name')
parser.add_argument('--db-root', default='/home/dgyoo/workspace/datain/ILSVRC', metavar='DIR', type=str,
                    help='root to the dataset')
parser.add_argument('--dst-dir', default='/home/dgyoo/workspace/dataout/dl-frame', metavar='DIR', type=str,
                    help='destination directory in that to save output data')
parser.add_argument('--arch', default='resnet18', metavar='NAME', type=str,
                    help='model architecture')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='if specified, use a pre-trained model')
parser.add_argument('--start-from', default='', metavar='PATH', type=str,
                    help='path to a model from that to resume training')
parser.add_argument('--num-worker', default=16, metavar='N', type=int,
                    help='number of workers that provide mini-batches')
parser.add_argument('--num-epoch', default=90, metavar='N', type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, metavar='N', type=int,
                    help='mini-batch size')
parser.add_argument('--learn-rate', default=0.1, metavar='LR', type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, metavar='M', type=float,
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, metavar='W', type=float,
                    help='weight decay')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='if specified, evaluate model on validation set')
ignore = ['db', 'db_root', 'dst_dir', 'arch', 'start_from', 'num_worker', 'num_epoch', 'evaluate']
opt = parser.parse_args(sys.argv[3:])

############################
# Class to create a model. #
############################
class Model():

    def __init__(self):
        self.model = None
        self.criterion = None
        self.optimizer = None

    def create_model(self):
        if opt.pretrained:
            print('Load ' + opt.arch + ' pre-trained on image-net.')
            model = models.__dict__[opt.arch](pretrained=True)
        else:
            print('Create ' + opt.arch + '.')
            model = models.__dict__[opt.arch]()
        if opt.arch.startswith('alexnet') or opt.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model)
        self.model = model.cuda()

    def create_criterion(self):
        self.criterion = nn.CrossEntropyLoss().cuda()

    def create_optimizer(self):
        self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                opt.learn_rate,
                momentum=opt.momentum,
                weight_decay=opt.weight_decay)

########################
# Class to build a db. #
########################
class Db():

    def __init__(self):
        self.train = None
        self.val = None
        self.stats = None

    def _make_dataset(self, split):
        assert split in ['train', 'val']
        path_db = os.path.join(opt.dst_dir_db, 'db_{}.pth'.format(split))
        try:
            print('Load {} db.'.format(split))
            db = torch.load(path_db)
        except:
            print('Make {} db.'.format(split))
            db = importlib.import_module(opt.db)
            db = db.make_dataset(opt.db_root, split)
            assert len(db['pairs']) > 0
            os.makedirs(opt.dst_dir_db, exist_ok=True)
            torch.save(db, path_db)
        return db

    def build(self):
        self.train = self._make_dataset('train')
        self.val = self._make_dataset('val')

    def estimate_stats(self, num_sample=1e5):
        path_stats = os.path.join(opt.dst_dir_db, 'stats.pth')
        try:
            print('Load db stats.')
            stats = torch.load(path_stats)
        except:
            print('Estimate db stats.')
            num_sample = min(num_sample, len(self.train['pairs']))
            num_batch = int(-(-num_sample // opt.batch_size))
            batch_manager = BatchManagerTrain(self.train)
            mean_meter = utils.AverageMeter()
            std_meter = utils.AverageMeter()
            for i, (batch, _) in enumerate(batch_manager.get_loader()):
                mean = batch.mean(3).mean(2).mean(0)
                std = batch.view(opt.batch_size, 3, -1).std(2).mean(0)
                mean_meter.update(mean, batch.size(0))
                std_meter.update(std, batch.size(0))
                print('Batch {}/{}, mean {}, std {}'.format(
                    i + 1, num_batch,
                    utils.to_string(mean_meter.avg, '%.4f'),
                    utils.to_string(std_meter.avg, '%.4f')))
                if i + 1 == num_batch: break
            stats = {'mean': mean_meter.avg, 'std':std_meter.avg}
            os.makedirs(opt.dst_dir_db, exist_ok=True)
            torch.save(stats, path_stats)
        self.stats = stats

###############################
# Classes to provide batches. #
###############################
class BatchManagerTrain(Dataset):
    
    def __init__(self, db, stats=None):
        self.collate_fn = default_collate
        self.db = db
        self.stats = stats
    
    # Get batch.
    def __getitem__(self, index):
        images, targets = [], []
        db_size = len(self.db['pairs'])
        for batch_index in range(opt.batch_size):
            db_index = index * opt.batch_size + batch_index
            if db_index == db_size: break
            path, target = self.db['pairs'][db_index]
            with open(path, 'rb') as f:
                with Image.open(f) as image:
                    image = image.convert('RGB')
            image = t.RandomSizedCrop(224)(image)
            image = t.RandomHorizontalFlip()(image)
            image = t.ToTensor()(image)
            if not self.stats is None:
                image = t.Normalize(
                        mean=self.stats['mean'],
                        std=self.stats['std'])(image)
            images.append(image)
            targets.append(target)
        return self.collate_fn(images), self.collate_fn(targets)

    # Number of batches.
    def __len__(self):
        return -(-len(self.db['pairs']) // opt.batch_size)

    def get_loader(self):
        shuffle(self.db['pairs'])
        return _data_loader(self, opt.num_worker)

    def get_evaluator(self):
        return _evaluator

class BatchManagerVal(Dataset):
    
    def __init__(self, db, stats=None):
        self.collate_fn = default_collate
        self.db = db
        self.stats = stats
        shuffle(self.db['pairs'])
    
    # Get batch.
    def __getitem__(self, index):
        images, targets = [], []
        db_size = len(self.db['pairs'])
        for batch_index in range(opt.batch_size):
            db_index = index * opt.batch_size + batch_index
            if db_index == db_size: break
            path, target = self.db['pairs'][db_index]
            with open(path, 'rb') as f:
                with Image.open(f) as image:
                    image = image.convert('RGB')
            image = t.Scale(256)(image)
            image = t.CenterCrop(224)(image)
            image = t.ToTensor()(image)
            if not self.stats is None:
                image = t.Normalize(
                        mean=self.stats['mean'],
                        std=self.stats['std'])(image)
            images.append(image)
            targets.append(target)
        return self.collate_fn(images), self.collate_fn(targets)

    # Number of batches.
    def __len__(self):
        return -(-len(self.db['pairs']) // opt.batch_size)

    def get_loader(self):
        return _data_loader(self, opt.num_worker)

    def get_evaluator(self):
        return _evaluator

def _data_loader(batch_provider, num_worker):
    return DataLoader(
            batch_provider,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=num_worker,
            collate_fn=lambda x:x[0],
            pin_memory=True,
            drop_last=False)

def _evaluator(outputs, targets):
    return metric.accuracy(outputs, targets, topk=(1, 5))
