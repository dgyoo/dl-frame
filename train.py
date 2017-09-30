import torch
import time

import utils

def train(loader, model, criterion, optimizer, evaluator, epoch):
    
    # Initialize meters.
    data_time = utils.AverageMeter()
    net_time = utils.AverageMeter()
    loss_meter = utils.AverageMeter()
    eval_meter = utils.AverageMeter()
    
    # Do the job.
    print('Start training at epoch {}.'.format(epoch))
    model.train()
    t0 = time.time()
    for i, (inputs, targets) in enumerate(loader):
        
        # Set variables.
        targets = targets.cuda(async=True)
        inputs_var = torch.autograd.Variable(inputs)
        targets_var = torch.autograd.Variable(targets)

        # Measure data time.
        data_time.update(time.time() - t0)
        t0 = time.time()
        
        # Forward.
        outputs = model(inputs_var)
        loss = criterion(outputs, targets_var)
        evals = evaluator(outputs.data, targets)

        # Backward.
        optimizer.zero_grad()
        loss.backward()

        # Update.
        optimizer.step()

        # Accumulate statistics.
        loss_meter.update(loss.data[0], targets.size(0))
        eval_meter.update(evals, targets.size(0))

        # Measure network time.
        net_time.update(time.time() - t0)
        t0 = time.time()

        # Print iteration.
        print('Epoch {0} Batch {1}/{2} '
                'T-data {data_time.val:.2f} ({data_time.avg:.2f}) '
                'T-net {net_time.val:.2f} ({net_time.avg:.2f}) '
                'Loss {loss.val:.2f} ({loss.avg:.2f}) '
                'Eval {eval_val} ({eval_avg})'.format(
                    epoch, i + 1, len(loader),
                    data_time=data_time,
                    net_time=net_time,
                    loss=loss_meter,
                    eval_val=utils.to_string(eval_meter.val),
                    eval_avg=utils.to_string(eval_meter.avg)))

    # Summerize results.
    print('Summary of training at epoch {epoch:d}.\n'
            '  Number of pairs: {num_sample:d}\n'
            '  Number of batches: {num_batch:d}\n'
            '  Total time for data: {data_time:.2f} sec\n'
            '  Total time for network: {net_time:.2f} sec\n'
            '  Total time: {total_time:.2f} sec\n'
            '  Average loss: {avg_loss:.4f}\n'
            '  Performance: {avg_perf}\n'.format(
                epoch=epoch,
                num_sample=loss_meter.count,
                num_batch=len(loader),
                data_time=data_time.sum,
                net_time=net_time.sum,
                total_time=data_time.sum+net_time.sum,
                avg_loss=loss_meter.avg,
                avg_perf=utils.to_string(eval_meter.avg, '%.4f')))
