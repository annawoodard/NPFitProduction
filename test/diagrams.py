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

import numpy as np

from EffectiveTTVProduction.EffectiveTTVProduction.cross_sections import *

parser = argparse.ArgumentParser(description='add Higgs Effective Lagrangian model to gridpack')
parser.add_argument('gridpack', help='gridpack to clone')
parser.add_argument('madgraph', help='madgraph directory with model')
parser.add_argument('inparams', help='param card to clone')
parser.add_argument('outparams', help='param card to update')
parser.add_argument('cores', help='number of cores to use')
parser.add_argument('events', type=int, help='number of events to use for cross section calculation')
parser.add_argument('coefficient', type=str, help='coefficient to scan')
parser.add_argument('process', help='process card to run')
parser.add_argument('value', type=float, help='value to set the coefficient at')

args = parser.parse_args()

get_cross_section(args, args.process, args.value)
