import os
import sys
import shutil
import argparse
import importlib

import torch
import torch.backends.cudnn as cudnn

import utils
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

    # Set destimation directories.
    changes = utils.arg_changes(task.parser, opt, task.ignore)
    opt.dst_dir_db = os.path.join(opt.dst_dir, opt.db)
    dst_dir_model = os.path.join(opt.dst_dir_db, opt.arch)
    if opt.start_from:
        assert opt.start_from.endswith('.pth.tar')
        dst_dir_model = opt.start_from[:-8]
    if changes: dst_dir_model += ',' + changes
    dst_path_model = os.path.join(dst_dir_model, '{:03d}.pth.tar')

    # Create loggers.
    logger_train = utils.Logger(os.path.join(dst_dir_model, 'train.log'))
    logger_val = utils.Logger(os.path.join(dst_dir_model, 'val.log'))
    assert len(logger_train) == len(logger_val)

    # Define start model, epoch, and the best performance, to resume train.
    start_epoch = len(logger_train)
    best_perform = logger_val.max() if start_epoch > 0 else 0
    start_from = dst_path_model.format(start_epoch) if start_epoch > 0 else opt.start_from

    # Create model, criterion, optimizer.
    model = task.Model()
    model.create_model()
    model.create_criterion()
    model.create_optimizer()

    # Load parameters if necessary.
    if start_from:
        print('Load a model from that to resume training.\n'
                '({})'.format(start_from))
        checkpoint = torch.load(start_from)
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
                batch_manager_val.get_evaluator())
        return

    # Do the job.
    cudnn.benchmark = True
    os.makedirs(dst_dir_model, exist_ok=True)
    for epoch in range(start_epoch, opt.num_epoch):
        
        # Adjust learning rate.
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = opt.learn_rate * (.1 ** (epoch // 30))

        # Train.
        print('Start training at epoch {}.'.format(epoch))
        train.train(
                batch_manager_train.get_loader(),
                model.model,
                model.criterion,
                model.optimizer,
                batch_manager_train.get_evaluator(),
                logger_train,
                epoch + 1)

        # Val.
        print('Start validation at epoch {}.'.format(epoch))
        perform = val.val(
                batch_manager_val.get_loader(),
                model.model,
                model.criterion,
                batch_manager_val.get_evaluator(),
                logger=logger_val,
                epoch=epoch + 1)

        # Save model.
        print('Save this model.')
        data = {
            'opt': opt,
            'log_train': logger_train.read(),
            'log_val': logger_val.read(),
            'state_dict': model.model.state_dict(),
            'optimizer' : model.optimizer.state_dict()}
        torch.save(data, dst_path_model.format(epoch + 1))

        # Remove previous model.
        if epoch > 0:
            os.system('rm {}'.format(dst_path_model.format(epoch)))

        # Backup the best model.
        if perform > best_perform:
            print('Save this model as the best.')
            os.system('cp {} {}'.format(
                dst_path_model.format(epoch + 1),
                os.path.join(dst_dir_model, 'best.pth.tar')))
        print('')

if __name__ == '__main__':
    main()
