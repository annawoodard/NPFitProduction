"""
Given low and high limits on the Wilson coefficients c_j, calculate
cross sections for a grid of points with dimensionality equal to the
number of coeffients, and each axis spanning low < c_j < high.

"""
import argparse
import sys

import numpy as np

from EffectiveTTVProduction.EffectiveTTVProduction.cross_sections import CrossSectionScan, get_cross_section
from EffectiveTTVProduction.EffectiveTTVProduction.utils import cartesian_product


parser = argparse.ArgumentParser(description='calculate cross sections')
parser.add_argument('numvalues', type=int, help='number of values to scan per coefficient')
parser.add_argument('cores', type=int, help='number of cores to use')
parser.add_argument('coefficients', type=str, help='comma-delimited list of wilson coefficients to scan')
parser.add_argument('events', type=int, help='number of events to use for cross section calculation')
parser.add_argument('madgraph', type=str, help='tarball containing madgraph')
parser.add_argument('np_model', type=str, help='tarball containing NP model')
parser.add_argument('np_param_path', type=str,
                    help='path (relative to the unpacked madgraph tarball) to the NP parameter card')
parser.add_argument('cards', type=str,
                    help='path to the cards directory (must contain run_card.dat, grid_card.dat, '
                    'me5_configuration.txt and the parameter card pointed to by np_param_path)')
parser.add_argument('low', type=float, help='lowest coefficient value to consider')
parser.add_argument('high', type=float, help='highest coefficient value to consider')
parser.add_argument('process_card', type=str, help='which process card to run')
parser.add_argument('indices', type=int, nargs='+', help='the indices of points to calculate')

args = parser.parse_args()
args.coefficients = args.coefficients.split(',')
process = args.process_card.split('/')[-1].replace('.dat', '')

values = [np.hstack([np.zeros(1), np.linspace(args.low, args.high, args.numvalues - 1, endpoint=True)]) for c in args.coefficients]
points = cartesian_product(*values)
print points
print values
print len(points)
result = CrossSectionScan()

for attempt in range(5):
    # Sometimes MG can fail if the Wilson coefficient values are too large.
    # If this happens, try again a few times with smaller values.
    try:
        for i in args.indices:
            point = points[i]
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
            result.add(point, cross_section, process, args.coefficients)
        break
    except RuntimeError as e:
        print '{}: halving coefficient values and trying again'.format(e)
        points = points / 2.

if len(result.points) is 0:
    print 'failed to calculate any points'
    sys.exit(1)
else:
    result.dump('cross_sections.npz')
