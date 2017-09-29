from collections import Iterable

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0 and isinstance(val, Iterable):
            self.sum = [0 for _ in val]
            self.avg = [0 for _ in val]
        self.val = val
        self.count += n
        if isinstance(val, Iterable):
            for i, v in enumerate(val):
                self.sum[i] += v * n
                self.avg[i] = self.sum[i] / self.count
        else:
            self.sum += val * n
            self.avg = self.sum / self.count

def to_string(values, precision='%.2f'):
    if not isinstance(values, Iterable):
        return precision % v
    string = ''
    for v in values:
        string += (precision % v) + ','
    return string[:-1]
