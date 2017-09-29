import os
import sys
import shutil
import argparse
import importlib

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

import train, val

sys.path.append('./db')
sys.path.append('./task')

def main():

    # Define task.
    parser = argparse.ArgumentParser(
            description='Large-scale deep learning framework.')
    parser.add_argument('--task', metavar='NAME', type=str, 
            help='specify a task name that defined in $ROOT/task/')
    arg = parser.parse_args(sys.argv[1:3])
    task = importlib.import_module(arg.task)
    opt = task.opt

    # Print options.
    print('Options.')
    for k, v in opt.__dict__.items():
        print('  {0}: {1}'.format(k, v))

    # Create model, criterion, optimizer.
    model = task.create_model()
    criterion = task.create_criterion()
    optimizer = task.create_optimizer(model)

    # Load parameters if necessary.
    best_perform = 0
    if opt.start_from:
        print('Start from ' + opt.start_from)
        checkpoint = torch.load(opt.start_from)
        opt.start_epoch = checkpoint['epoch']
        best_perform = checkpoint['best_perform']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Create data loader.
    train_loader = data_loader(
            task.BatchProviderTrain(),
            opt.num_worker)
    val_loader = data_loader(
            task.BatchProviderVal(),
            opt.num_worker)
    
    # Evaluate the model and exit for evaluation mode.
    if opt.evaluate:
        val.val(
                val_loader,
                model,
                criterion,
                task.evaluate_batch_val,
                0)
        return

    # Do the job.
    cudnn.benchmark = True
    for epoch in range(opt.start_epoch, opt.num_epoch):
        
        # Adjust learning rate.
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.learn_rate * (.1 ** (epoch // 30))

        # Train.
        train.train(
                train_loader,
                model,
                criterion,
                optimizer,
                task.evaluate_batch_train,
                epoch + 1)

        # Val.
        perform = val.val(
                val_loader,
                model,
                criterion,
                task.evaluate_batch_val,
                epoch + 1)

        # Save checkpoint.
        best_perform = max(perform, best_perform)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'best_perform': best_perform,
            'optimizer' : optimizer.state_dict()},
            perform > best_perform)

def data_loader(batch_provider, num_worker):
    return torch.utils.data.DataLoader(
            batch_provider,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=num_worker,
            collate_fn=lambda x:x[0],
            pin_memory=True,
            drop_last=False)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    main()
