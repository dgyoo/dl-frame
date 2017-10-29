import os
import argparse

import utils


def main():
    parser = argparse.ArgumentParser(description='Summarize training results')
    parser.add_argument('--root', default='/home/dgyoo/workspace/dataout/dl-frame', metavar='DIR', type=str,
                        help='root to the log files')
    parser.add_argument('--target-index', default=2, metavar='N', type=int,
                        help='target evaluation metric index in the log files (a column index of a log file)')
    parser.add_argument('--decimal-places', default=2, metavar='N', type=int,
                        help='decimal places')
    args = parser.parse_args()

    # Find log files and directories.
    log_dirs = []
    for root, _, files in os.walk(args.root):
        for f in files:
            if f.endswith('.log'):
                log_dirs.append(root)
    log_dirs = [log_dir.replace(',', '~') for log_dir in log_dirs]
    log_dirs = sorted(list(set(log_dirs)))
    log_dirs = [log_dir.replace('~', ',') for log_dir in log_dirs]

    # Find the base directory.
    splits = log_dirs[0].split(os.sep)
    for i in range(len(splits)):
        if not all([os.path.join(*splits[:i + 1]) in log_dir for log_dir in log_dirs]):
            break
    base_dir = os.path.join(*splits[:i])
    base_dir = log_dirs[0][:log_dirs[0].find(base_dir) + len(base_dir) + 1]

    # Do the job.
    skips = []
    form = '%.{:d}f'.format(args.decimal_places)
    print('Summerizing results in {}'.format(base_dir))
    for log_dir in log_dirs:
        logger_train = utils.Logger(os.path.join(log_dir, 'train.log'))
        logger_val = utils.Logger(os.path.join(log_dir, 'val.log'))
        if len(logger_train) != len(logger_val):
            skips.append(log_dir)
            continue
        log_train = logger_train.read()
        log_val = logger_val.read()
        targets = [log[args.target_index] for log in log_val]
        index = targets.index(max(targets))
        print('E {:02d}/{:02d} | TL {} TE {} | VL {} VE {} | {}'.format(
            int(log_train[index][0]),
            len(logger_train),
            utils.to_string(log_train[index][1], form),
            utils.to_string(log_train[index][2:], form),
            utils.to_string(log_val[index][1], form),
            utils.to_string(log_val[index][2:], form),
            log_dir[len(base_dir):]))

    for skip in skips:
        print('Skip {} since len(train) != len(val)'.format(skip[len(base_dir):]))

if __name__ == '__main__':
    main()
