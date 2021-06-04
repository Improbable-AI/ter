import csv
import os

class CsvLogger(object):
    def __init__(self, filename, keys):
        assert filename is not None
        self.f = open(filename, "wt")
        self.logger = csv.DictWriter(self.f, fieldnames=keys)
        self.logger.writeheader()
        self.f.flush()

    def write_row(self, info):
        self.logger.writerow(info)
        self.f.flush()