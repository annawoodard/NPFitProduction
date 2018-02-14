import imp
import os
import subprocess
import shutil

import numpy as np

cross_sections = imp.load_source('', '/afs/crc.nd.edu/user/a/awoodard/releases/np-fit-production/CMSSW_7_4_7/src/NPFitProduction/NPFitProduction/python/cross_sections.py')
if not os.path.isdir('cards/ttV/1'):
    shutil.copytree('/afs/crc.nd.edu/user/a/awoodard/releases/np-fit-production/CMSSW_7_4_7/src/NPFitProduction/NPFitProduction/test/cards/ttV/1', 'cards/ttV/1')

if not os.path.isdir('cards/ttV/1'):
    shutil.copytree('/afs/crc.nd.edu/user/a/awoodard/releases/np-fit-production/CMSSW_7_4_7/src/NPFitProduction/NPFitProduction/test/cards/ttV/1', 'cards/ttV/1')

scan = cross_sections.CrossSectionScan('/afs/crc.nd.edu/user/a/awoodard/np-fit-production/data/8d.npz')
# coefficients = ['cuB']
coefficients = ['cuW', 'cuB', 'cH', 'tc3G', 'c3G', 'cHu', 'c2G', 'cuG']
# coefficients = ['cuB', 'cuG', 'c3G', 'cHu', 'tc3G', 'cH', 'c2G']
scale = 5.
tag = 'reweight_v25'
events = 100000
points = 600

num_sampled_points = 1000000
mins, maxes = cross_sections.get_bounds(coefficients, scan, scale)
sampled_points = np.zeros((num_sampled_points, len(coefficients)))
for i in range(num_sampled_points):
    point = []
    for column, coefficient in enumerate(coefficients):
        point += [np.random.uniform(mins[column], maxes[column])]
    sampled_points[i] = np.array([point])

scales = scan.evaluate(coefficients, sampled_points, 'ttZ').ravel()
window = (scales > 2) & (scales < 5)
sampled_points = sampled_points[window]
scales = scales[window]

sort = scales.argsort()
starting_point = sampled_points[sort][-1]

# starting_point = np.array([maxes[i] for i in range(len(coefficients))])
# # add noise so no two points are the same, or
# # MG will combine them (hack, should be done more carefully)
# starting_point -= starting_point / 100 * np.random.rand(len(coefficients))
# starting_point /= 2.
# for k, v in mins.items():
#     mins[k] = maxes[k] / 1e6
for i, c in enumerate(coefficients):
    mins[c] = sampled_points[:,i].min()

cross_sections.setup_sandbox(
    'MG5_aMC_v2_3_3.tar.gz',
    'HEL_UFO.third_gen.tar.gz',
    'HEL_UFO/restrict_no_b_mass.dat',
    coefficients,
    'process_cards/ttZ.dat',
    32,
    events,
    'cards/ttV/1',
    starting_point,
    tag,
)

with open('processtmp/Cards/me5_configuration.txt', 'a') as f:
    f.write("mg5_path = ../")

cross_sections.write_reweight_card(
    'processtmp/Cards/param_card.dat',
    'processtmp/Cards/reweight_card.dat',
    points,
    coefficients,
    mins,
    maxes
)
os.chdir('processtmp')
print subprocess.check_output(['./bin/generate_events', '-f'])

# points, weights = cross_sections.parse_lhe_weights('Events/reweight=ON/unweighted_events.lhe')
# constants, _, _, _ = np.linalg.lstsq(scan.model(points[::2]), weights[::2])
# predicted = np.dot(scan.model(points[1::2]), constants)

# for a, b in zip(weights[1::2], predicted):
#     print a, b

