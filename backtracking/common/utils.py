import logging
import random
import time
import numpy as np
from contextlib import contextmanager

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color='green', bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class RunningStat(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.mean = 0
        self.max = -np.inf
        self.min = np.inf
        self.history = list()

    def update(self, new_val):
        self.history.append(new_val)
        old_count = self.count
        new_count = self.count + 1
        self.mean = self.mean * (old_count / new_count) + new_val / new_count
        self.count = new_count
        self.max = max(self.max, new_val)
        self.min = min(self.min, new_val)

    def get_dict(self, header=None):
        return {'{}(max)'.format(header): self.max,
                '{}(min)'.format(header): self.min,
                '{}(mean)'.format(header): self.mean,
                '{}(median)'.format(header): np.median(self.history)}

    def __str__(self):
        return 'M:{:.4f}/m:{:.4f}/mu:{:.4f}'.format(self.max, self.min, self.mean)

class IntRunningStat(RunningStat):

    def __str__(self):
        if self.count > 0:
            max_val = int(self.max)
            min_val = int(self.min)
        else:
            max_val = min_val = 'None'
        return 'M:{}/m:{}/mu:{:.1f}'.format(max_val, min_val, self.mean)

_time_stats = dict()

@contextmanager
def timed(msg, key=None, vis=True):
    # To disable/enable all profiling by a flag
    if not vis:
        yield
        return

    print(colorize(msg, color='green'))
    tstart = time.time()
    yield
    dt = time.time() - tstart
    print(colorize("done in %.3f seconds"% dt, color='magenta'))
    if key is not None:
        if key not in _time_stats:
            _time_stats[key] = []            
        stats = _time_stats[key]
        stats.append(dt)
        print('\tOp: {}'.format(key))
        print('\t\t Max: {}'.format(max(stats)))
        print('\t\t Min: {}'.format(min(stats)))
        print('\t\t Mean: {}'.format(np.mean(stats)))
        print('\t\t n_calls: {}'.format(len(stats)))

@contextmanager
def timed_stat(stat):
    tstart = time.time()
    yield
    dt = time.time() - tstart
    stat.update(dt)

def col_print(lines, term_width=80, indent=0, pad=2):
  n_lines = len(lines)
  if n_lines == 0:
    return

  col_width = max(len(line) for line in lines)
  n_cols = int((term_width + pad - indent)/(col_width + pad))
  n_cols = min(n_lines, max(1, n_cols))

  col_len = int(n_lines/n_cols) + (0 if n_lines % n_cols == 0 else 1)
  if (n_cols - 1) * col_len >= n_lines:
    n_cols -= 1

  cols = [lines[i*col_len : i*col_len + col_len] for i in range(n_cols)]

  rows = list(zip(*cols))
  rows_missed = zip(*[col[len(rows):] for col in cols[:-1]])
  rows.extend(rows_missed)

  for row in rows:
    print(" "*indent + (" "*pad).join(line.ljust(col_width) for line in row))
