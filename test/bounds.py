import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempdir

import numpy as np

from EffectiveTTVProduction.EffectiveTTVProduction.cross_sections import *

parser = argparse.ArgumentParser(description='add Higgs Effective Lagrangian model to gridpack')
parser.add_argument('gridpack', help='gridpack to clone')
parser.add_argument('madgraph', help='madgraph directory with model')
parser.add_argument('inparams', help='param card to clone')
parser.add_argument('outparams', help='param card to update')
parser.add_argument('cores', help='number of cores to use')
parser.add_argument('events', type=int, help='number of events to use for cross section calculation')
parser.add_argument('left', type=float, help='lowest coefficient value to consider')
parser.add_argument('right', type=float, help='highest coefficient value to consider')
parser.add_argument('scale', type=float, help='maximum scaling to constrain coefficient values')
parser.add_argument('threshold', type=float, help='threshold at which to consider scale matched')
parser.add_argument('constraints', nargs='+', help='process cards to constrain scan range')
parser.add_argument('coefficient', type=str, help='coefficient to calculate bounds for')

args = parser.parse_args()

sm = mp_map([(get_cross_section, [args, c, 0.], {}) for c in args.constraints])
print 'sm is ', sm

# must be one at SM (NP=0)
left = bisect(args, sm, args.left, args.right, left_to_right=True)
right = bisect(args, sm, args.left, args.right, left_to_right=False)
print('left is '+str(left))
# high = bisect(args, sm, 0., float(args.high), lambda x: x > args.scale) if within_bounds(args, sm, args.high) else args.high
print('right is '+str(right))

res = {
    'left': left if left else args.left,
    'right': right if right else args.right,
    'args': args
}

np.save('bounds.npy', res)

