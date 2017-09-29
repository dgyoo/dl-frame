import sys
import argparse
import importlib
from random import shuffle
from PIL import Image

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.transforms as t
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

##########################
# Task-specific options. #
##########################
parser = argparse.ArgumentParser(description='Large-scale image classification')
parser.add_argument('--db', default='imagenet', metavar='NAME', type=str,
                    help='dataset name')
parser.add_argument('--db-root', default='/home/dgyoo/workspace/datain/ILSVRC', metavar='DIR', type=str,
                    help='root to the dataset')
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
parser.add_argument('--start-epoch', default=0, metavar='N', type=int,
                    help='starting epoch number to resume training')
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
opt = parser.parse_args(sys.argv[3:])

############################################
# Functions to define model, loss, solver. #
############################################
def create_model():
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
    return model.cuda()

def create_criterion():
    return nn.CrossEntropyLoss().cuda()

def create_optimizer(model):
    return torch.optim.SGD(
            model.parameters(),
            opt.learn_rate,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay)

###############################
# Classes to provide batches. #
###############################
class BatchProviderTrain(Dataset):
    
    def __init__(self):
        self.collate_fn = default_collate
        db = importlib.import_module(opt.db)
        self.pairs, self.classes = db.make_dataset(opt.db_root, 'train')
        shuffle(self.pairs)
        assert len(self.pairs) > 0
    
    # Get batch.
    def __getitem__(self, index):
        images, targets = [], []
        db_size = len(self.pairs)
        for batch_index in range(opt.batch_size):
            db_index = index * opt.batch_size + batch_index
            if db_index == db_size: break
            path, target = self.pairs[db_index]
            with open(path, 'rb') as f:
                with Image.open(f) as image:
                    image = image.convert('RGB')
            image = t.RandomSizedCrop(224)(image)
            image = t.RandomHorizontalFlip()(image)
            image = t.ToTensor()(image)
            image = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            images.append(image)
            targets.append(target)
        return self.collate_fn(images), self.collate_fn(targets)

    # Number of batches.
    def __len__(self):
        return -(-len(self.pairs) // opt.batch_size)

class BatchProviderVal(Dataset):
    
    def __init__(self):
        self.collate_fn = default_collate
        db = importlib.import_module(opt.db)
        self.pairs, self.classes = db.make_dataset(opt.db_root, 'val')
        assert len(self.pairs) > 0 and len(self.classes) > 0
    
    # Get batch.
    def __getitem__(self, index):
        images, targets = [], []
        db_size = len(self.pairs)
        for batch_index in range(opt.batch_size):
            db_index = index * opt.batch_size + batch_index
            if db_index == db_size: break
            path, target = self.pairs[db_index]
            with open(path, 'rb') as f:
                with Image.open(f) as image:
                    image = image.convert('RGB')
            image = t.Scale(256)(image)
            image = t.CenterCrop(224)(image)
            image = t.ToTensor()(image)
            image = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            images.append(image)
            targets.append(target)
        return self.collate_fn(images), self.collate_fn(targets)

    # Number of batches.
    def __len__(self):
        return -(-len(self.pairs) // opt.batch_size)

#########################################
# Functions to evaluate a result batch. #
#########################################
def evaluate_batch_train(outputs, targets):
    return _accuracy(outputs, targets, topk=(1, 5))

def evaluate_batch_val(outputs, targets):
    return _accuracy(outputs, targets, topk=(1, 5))

def _accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size)[0])
    return res
