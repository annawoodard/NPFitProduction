import argparse

import numpy as np

parser = argparse.ArgumentParser(description='merge cross sections')
parser.add_argument('outfile', help='name of output merged file')
parser.add_argument('infiles', nargs='+', help='files to merge')
args = parser.parse_args()


res = {}
for f in args.infiles:
    info = np.load(f.replace('file:', ''))[()]
    for process, cross_sections in info.items():
        try:
            res[process] = np.vstack([res[process], cross_sections])
        except KeyError:
            res[process] = cross_sections

np.save(args.outfile, res)
