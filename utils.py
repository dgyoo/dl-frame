from collections import Iterable

def arg_changes(argparser, arg, ignore=[]):
    arg_dict = vars(arg)
    changes_dict = {}
    for key in arg_dict:
        val0 = argparser.get_default(key)
        val = arg_dict[key]
        if val0 != val and not key in ignore:
            changes_dict[key] = val
    changes = ''
    for k in sorted(changes_dict):
        v = changes_dict[k]
        if isinstance(v, bool):
            v = int(v)
        elif isinstance(v, int):
            v = str(v)
        elif isinstance(v, float):
            v = '{:.2f}'.format(v)
        elif isinstance(v, str):
            pass
        else:
            raise Exception('Not supported argument value: {}'.format(v))
        changes += '{}={},'.format(k, v)
    return changes[:-1]

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
