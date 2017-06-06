"""
Multidimensional is hard, so implementing 1D first
It's more complicated than necessary because I want to make it easier to go multidimensional later
"""
from __future__ import print_function
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempdir
import time

import numpy as np

from EffectiveTTVProduction.EffectiveTTVProduction.operators import operators
from EffectiveTTVProduction.EffectiveTTVProduction.cross_sections import *

print('sys.argv {}'.format(str(sys.argv)))

parser = argparse.ArgumentParser(description='calculate cross sections')
parser.add_argument('numpoints', type=int, help='number of points to scan')
parser.add_argument('cores', type=int, help='number of cores to use')
parser.add_argument('events', type=int, help='number of events to use for cross section calculation')
parser.add_argument('process', type=str, help='which process card to run')
parser.add_argument('points', type=int, nargs='+', help='points to calculate')
parser.add_argument('bounds', type=str, help='bounds for calculation')
args = parser.parse_args()

info = np.load(args.bounds.replace('file:', ''))[()]
vars(info['args']).update(vars(args))
args = info['args']

process = args.process.split('/')[-1].replace('.dat', '')
dtype = [(name, 'f8') for name in operators]
np.linspace(info['low'], info['high'], num=args.numpoints)
values = np.hstack([np.array([0.0]), np.linspace(info['low'], info['high'], num=args.numpoints)])

for point in args.points:
    value = values[point]
    print('operators {}'.format(str(operators)))
    print('coefficient {}'.format(str(args.coefficient)))
    coefficients = np.array(tuple(0. if o != args.coefficient else value for o in operators), dtype=dtype)
    print(str(coefficients))
    cross_section = get_cross_section(args, args.process, value)
    row = np.array((coefficients, cross_section), dtype=[('coefficients', coefficients.dtype, coefficients.shape), ('cross section', 'f8')])

    try:
        cross_sections = np.vstack([cross_sections, row])
    except NameError:
        cross_sections = row

np.save('cross_sections.npy', {process: cross_sections})
