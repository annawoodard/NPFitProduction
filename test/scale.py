import argparse
import sys

import numpy as np

from NPFitProduction.NPFitProduction.cross_sections import CrossSectionScan, get_cross_section, get_bounds
from NPFitProduction.NPFitProduction.utils import cartesian_product

parser = argparse.ArgumentParser(description='calculate cross sections')

parser.add_argument('interpolate_numvalues', type=int, help='number of values to scan per coefficient')
parser.add_argument('calculate_numvalues', type=int, help='number of values to scan per coefficient')
parser.add_argument('cores', type=int, help='number of cores to use')
parser.add_argument('events', type=int, help='number of events to use for cross section calculation')
parser.add_argument('madgraph', type=str, help='tarball containing madgraph')
parser.add_argument('np_model', type=str, help='tarball containing NP model')
parser.add_argument('np_param_path', type=str,
                    help='path (relative to the unpacked madgraph tarball) to the NP parameter card')
parser.add_argument('cards', type=str,
                    help='path to the cards directory (must contain run_card.dat, grid_card.dat, '
                    'me5_configuration.txt and the parameter card pointed to by np_param_path)')
parser.add_argument('scale', type=float, help='maximum scaling to constrain coefficient values')
parser.add_argument('coefficients', type=str, help='comma-delimited list of wilson coefficients to scan')
parser.add_argument('process_card', type=str, help='which process card to run')
parser.add_argument('indices', type=int, nargs='+', help='the indices of points to calculate')
parser.add_argument('scan', type=str, help='coarse-grained scan points to constrain coefficient values')
args = parser.parse_args()

args.coefficients = tuple(args.coefficients.split(','))
process = args.process_card.split('/')[-1].replace('.dat', '')
coarse_scan = CrossSectionScan([args.scan.replace('file:', '')])
result = CrossSectionScan()

try:
    mins, maxes = get_bounds(args.coefficients, coarse_scan, args.scale, args.interpolate_numvalues)
except RuntimeError:
    raise

for i, value in enumerate(args.indices):
    if value == 0:
        # we must always include the SM point in order to calculate the scaling
        point = [0.0] * len(args.coefficients)
    else:
        try:
            point = []
            for column, coefficient in enumerate(args.coefficients):
                point += [np.random.uniform(mins[column], maxes[column])]
            cross_section = get_cross_section(
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
            result.add(point, np.array([cross_section]), process, args.coefficients)
        except RuntimeError as e:
            print e
            sys.exit(42)

result.dump('cross_sections.npz')
