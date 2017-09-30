import os
import sys
import shutil
import argparse
import importlib

import torch
import torch.backends.cudnn as cudnn

import train, val

sys.path.append('./db')
sys.path.append('./task')

def main():

    # Define task.
    parser = argparse.ArgumentParser(
            description='Large-scale deep learning framework.')
    parser.add_argument('--task', metavar='NAME', type=str, required=True,
            help='specify a task name that defined in $ROOT/task/')
    arg = parser.parse_args(sys.argv[1:3])
    task = importlib.import_module(arg.task)
    opt = task.opt

    # Print options.
    print('Options.')
    for k in sorted(vars(opt)):
        if not k.startswith('dst_dir'):
            print('  {0}: {1}'.format(k, opt.__dict__[k]))

    # Create model, criterion, optimizer.
    model = task.Model()
    model.create_model()
    model.create_criterion()
    model.create_optimizer()

    # Load parameters if necessary.
    best_perform = 0
    if opt.start_from:
        print('Start from ' + opt.start_from)
        checkpoint = torch.load(opt.start_from)
        opt.start_epoch = checkpoint['epoch']
        best_perform = checkpoint['best_perform']
        model.model.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])

    # Build db.
    db = task.Db()
    db.build()
    db.estimate_stats()

    # Create batch manager..
    batch_manager_train = task.BatchManagerTrain(db.train, db.stats)
    batch_manager_val = task.BatchManagerVal(db.val, db.stats)
    
    # Evaluate the model and exit for evaluation mode.
    if opt.evaluate:
        val.val(
                batch_manager_val.get_loader(),
                model.model,
                model.criterion,
                task.evaluate_batch_val)
        return

    # Do the job.
    cudnn.benchmark = True
    for epoch in range(opt.start_epoch, opt.num_epoch):
        
        # Adjust learning rate.
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = opt.learn_rate * (.1 ** (epoch // 30))

        # Train.
        train.train(
                batch_manager_train.get_loader(),
                model.model,
                model.criterion,
                model.optimizer,
                batch_manager_train.get_evaluator(),
                epoch + 1)

        # Val.
        perform = val.val(
                batch_manager_val.get_loader(),
                model.model,
                model.criterion,
                batch_manager_val.get_evaluator(),
                epoch + 1)

        # Save checkpoint.
        best_perform = max(perform, best_perform)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.model.state_dict(),
            'best_perform': best_perform,
            'optimizer' : model.optimizer.state_dict()},
            perform > best_perform)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    main()
