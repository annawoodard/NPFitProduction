"""
Given low and high limits on the Wilson coefficients c_j, produce
gridpacks for a grid of points with dimensionality equal to the
number of coeffients, and each axis spanning low < c_j < high.

"""
from __future__ import print_function
import argparse
import os
import re
import shutil
import subprocess
import tempdir

import numpy as np

from EffectiveTTVProduction.EffectiveTTVProduction.cross_sections import CrossSectionScan, get_points, setup_model
from EffectiveTTVProduction.EffectiveTTVProduction.utils import cartesian_product

parser = argparse.ArgumentParser(description='produce gridpacks')

parser.add_argument('numvalues', type=int, help='number of values to scan per coefficient')
parser.add_argument('cores', type=int, help='number of cores to use')
parser.add_argument('coefficients', type=str, help='comma-delimited list of wilson coefficients to scan')
parser.add_argument('events', type=int, help='number of events to use for cross section calculation')
parser.add_argument('sm_gridpack', type=str, help='tarball containing an SM gridpack')
parser.add_argument('madgraph', type=str, help='tarball containing madgraph')
parser.add_argument('np_model', type=str, help='tarball containing NP model')
parser.add_argument('np_param_path', type=str,
                    help='path (relative to the unpacked madgraph tarball) to the NP parameter card')
parser.add_argument('cards', type=str,
                    help='path to the cards directory (must contain run_card.dat, grid_card.dat, '
                    'me5_configuration.txt and the parameter card pointed to by np_param_path)')
parser.add_argument('process_card', type=str, help='which process card to run')
parser.add_argument('--scale', type=float, help='maximum scaling to constrain coefficient values')
parser.add_argument('--scan', type=str,
                    help='coarse-grained scan point file-- note: either (scale, scan and constraints) or '
                    '(low and high) are required options')
parser.add_argument('--constraints', help='comma delimited list of processes to include for range finding')
parser.add_argument('--low', type=float, help='lower bound of coefficient range')
parser.add_argument('--high', type=float, help='upper bound of coefficient range')
parser.add_argument('index', type=int, help='the index of the point to calculate')
args = parser.parse_args()

args.coefficients = args.coefficients.split(',')
process = args.process_card.split('/')[-1].replace('.dat', '')

if args.scan and args.scale and args.constraints:
    coarse_scan = CrossSectionScan([args.scan.replace('file:', '')])
    coarse_scan.prune(args.constraints)
    points = get_points(args.coefficients, coarse_scan, args.scale, args.numvalues)
elif args.low and args.high:
    values = [np.hstack([np.zeros(1), np.linspace(args.low, args.high, args.numvalues)]) for c in args.coefficients]
    points = cartesian_product(*values)
else:
    raise NotImplementedError('either scale and scan or interval are required')

point = points[args.index]

start = os.getcwd()
with tempdir.TempDir() as sandbox:
    os.chdir(sandbox)

    outdir = setup_model(
        start,
        args.madgraph,
        args.np_model,
        args.np_param_path,
        args.coefficients,
        args.process_card,
        args.cores,
        args.events,
        args.cards,
        point
    )
    carddir = os.path.join(outdir, 'Cards')

    with open(os.path.join(carddir, 'run_card.dat'), 'a') as f:
        print('.true. =  gridpack', file=f)

    output = subprocess.check_output(['./{}/bin/generate_events'.format(outdir), '-f'])
    m = re.search("Cross-section :\s*(.*) \+", output)
    cross_section = float(m.group(1)) if m else np.nan

    subprocess.call(['tar', 'xzf', '{}/run_01_gridpack.tar.gz'.format(outdir)])

    subprocess.call(['tar', 'xaf', os.path.join(start, args.sm_gridpack), 'mgbasedir'])
    subprocess.call(['tar', 'xaf', os.path.join(start, args.sm_gridpack), 'runcmsgrid.sh'])
    os.mkdir('process')
    shutil.move('madevent', 'process')
    shutil.move('run.sh', 'process')

    annotator = CrossSectionScan()
    annotator.add(point, cross_section, process, args.coefficients)
    annotator.dump('point.npz')

    subprocess.call(['tar', 'cJpsf', 'gridpack.tar.xz', 'mgbasedir', 'process', 'runcmsgrid.sh', 'point.npz'])
    shutil.move('gridpack.tar.xz', start)
