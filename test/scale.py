import argparse
import sys

import numpy as np

from EffectiveTTVProduction.EffectiveTTVProduction.cross_sections import CrossSectionScan, get_cross_section, get_points
from EffectiveTTVProduction.EffectiveTTVProduction.utils import cartesian_product

parser = argparse.ArgumentParser(description='calculate cross sections')

parser.add_argument('interpolate_numvalues', type=int, help='number of values to scan per coefficient')
parser.add_argument('calculate_numvalues', type=int, help='number of values to scan per coefficient')
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
parser.add_argument('scale', type=float, help='maximum scaling to constrain coefficient values')
parser.add_argument('process_card', type=str, help='which process card to run')
parser.add_argument('indices', type=int, nargs='+', help='the indices of points to calculate')
parser.add_argument('scan', type=str, help='coarse-grained scan points to constrain coefficient values')
args = parser.parse_args()

args.coefficients = tuple(args.coefficients.split(','))
process = args.process_card.split('/')[-1].replace('.dat', '')
coarse_scan = CrossSectionScan([args.scan.replace('file:', '')])
result = CrossSectionScan()

try:
    points = get_points(args.coefficients, coarse_scan, args.scale, args.interpolate_numvalues, args.calculate_numvalues)
except RuntimeError:
    # Something is wrong with the fit; zoom in and hope we do better next time
    ranges = None
    for process, points in coarse_scan.points[args.coefficients].items():
        endpoint = np.array([np.abs(points[:, i]).max() / 2. for i in range(len(args.coefficients))])
        if ranges is None:
            ranges = endpoint
        else:
            ranges = np.amin(np.vstack([endpoint, ranges]), axis=0)
    values = [np.hstack([np.zeros(1), np.linspace(-1. * ranges[i], ranges[i], args.calculate_numvalues - 1)]) for i in range(len(args.coefficients))]
    points = cartesian_product(*values)

coarse_points = coarse_scan.points[args.coefficients]
print 'points ', points
print 'points shape ', points.shape

for i in args.indices:
    print 'index ', i
    print 'row ', points[i]
    point = points[i]
    if process in coarse_points:
        common = np.where((coarse_points[process] == point).all(axis=1))[0]
        if len(common) > 0:
            #  we are already zoomed in, no need to calculate again
            cross_section = coarse_scan.cross_sections[args.coefficients][process][common[0]]
            result.add(points[i], np.array([cross_section]), process, args.coefficients)
            print 'skipping because point {} is already calculated: {}'.format(str(i), str(points[i]))
            continue
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
    result.add(points[i], np.array([cross_section]), process, args.coefficients)

result.dump('cross_sections.npz')
