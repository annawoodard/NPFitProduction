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

scan = cross_sections.CrossSectionScan('/hadoop/store/user/awoodard/ttV/cross_sections/19/2d/interval/cross_sections_97.npz')
coefficients = ['cuW', 'cuB', 'cH', 'tc3G', 'c3G', 'cHu', 'c2G', 'cuG']
scale = 4

_, maxes = cross_sections.get_bounds(coefficients, scan, scale, 1000)
# the starting point should be proportional to the max
starting_point = np.array([maxes[i] / 2. for i in range(len(coefficients))])
starting_point += starting_point / 10 * np.random.rand(len(coefficients))

cross_sections.setup_sandbox(
    'MG5_aMC_v2_3_3.tar.gz',
    'HEL_UFO.third_gen.tar.gz',
    'HEL_UFO/restrict_no_b_mass.dat',
    coefficients,
    'process_cards/ttZ.dat',
    32,
    50000,
    'cards/ttV/1',
    starting_point,
    'reweight_v5'
)

with open('processtmp/Cards/me5_configuration.txt', 'a') as f:
    f.write("mg5_path = ../")

cross_sections.write_reweight_card(
    'processtmp/Cards/param_card.dat',
    'processtmp/Cards/reweight_card.dat',
    600,
    ['cuW', 'cuB', 'cH', 'tc3G', 'c3G', 'cHu', 'c2G', 'cuG'],
    scan,
    scale
)
os.chdir('processtmp')
print subprocess.check_output(['./bin/generate_events', '-f'])

# points, weights = cross_sections.parse_lhe_weights('Events/reweight=ON/unweighted_events.lhe')
# constants, _, _, _ = np.linalg.lstsq(scan.model(points[::2]), weights[::2])
# predicted = np.dot(scan.model(points[1::2]), constants)

# for a, b in zip(weights[1::2], predicted):
#     print a, b

