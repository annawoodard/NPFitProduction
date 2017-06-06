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

from EffectiveTTVProduction.EffectiveTTVProduction.operators import operators

def get_cross_section(args, card, value, diagrams=False):
    print('card and value are {} {}'.format(card, str(value)))
    with tempdir.TempDir() as source:
        subprocess.call(['tar', 'xaf', args.gridpack, '--directory={0}'.format(source)])

        with open(os.path.join(source, args.inparams)) as f:
            inparams = f.readlines()

        with tempdir.TempDir() as madgraph:
            with tarfile.open(args.madgraph, "r:gz") as f:
                f.extractall(madgraph)

            with open(os.path.join(madgraph, args.outparams)) as f:
                outparams = f.readlines()

            with open(os.path.join(madgraph, args.outparams), 'w') as f:
                pattern = re.compile('(\d*) ([\de\+\-\.]*) (#.*) ')
                coefficients = []
                for operator in operators:
                    if operator == args.coefficient:
                        coefficients.append(value)
                    else:
                        coefficients.append(0.0)
                for out_line in outparams:
                    match = re.search(pattern, out_line)
                    if match:
                        out_id, out_value, out_label = match.groups()
                        for in_line in inparams:
                            match = re.search(pattern, in_line)
                            if match:
                                in_id, in_value, in_label = match.groups()
                                if in_label == out_label:
                                    out_line = re.sub(re.escape(out_value), in_value, out_line)
                    for operator, coefficient in zip(operators, coefficients):
                        out_line = re.sub('\d*.\d00000.*\# {0} '.format(operator), '{0} # {1}'.format(coefficient, operator), out_line)

                    f.write(out_line)


            subprocess.check_output(['python', os.path.join(madgraph, 'bin', 'mg5_aMC'), '-f', card])

            shutil.copy(os.path.join(source, 'process/madevent/Cards/run_card.dat'), 'processtmp/Cards')
            shutil.copy(os.path.join(source, 'process/madevent/Cards/grid_card.dat'), 'processtmp/Cards')
            shutil.copy(os.path.join(source, 'process/madevent/Cards/me5_configuration.txt'), 'processtmp/Cards')
            with open('processtmp/Cards/me5_configuration.txt', 'a') as f:
                print('run_mode = 2', file=f)
                print('nb_core = {0}'.format(args.cores), file=f)
                # print('lhapdf = /afs/crc.nd.edu/user/a/awoodard/local/bin/lhapdf-config', file=f)
                print('lhapdf = /cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/lhapdf/6.1.6/share/LHAPDF/../../bin/lhapdf-config', file=f)
                print('automatic_html_opening = False', file=f)


            with open('processtmp/Cards/run_card.dat', 'a') as f:
                print(' {} =  nevents'.format(args.events), file=f)
                print('.false. =  gridpack', file=f)

            output = subprocess.check_output(['./processtmp/bin/generate_events', '-f'])
            m = re.search("Cross-section :\s*(.*) \+", output)

            if diagrams:
                subprocess.call(['tar cJpsf diagrams.tar.xz processtmp/'], shell=True)

        shutil.rmtree('processtmp')

        print('returning '+str(float(m.group(1)) if m else 9999999))
        return float(m.group(1)) if m else 9999999.

def bisect(args, sm, low, high, go_left):
    print('calling bisect with sm {} low {} high {}'.format(str(sm), str(low), str(high)))
    scales = {}
    middle = (low + high) / 2.
    for card in args.constraints:
        scales[card] = get_cross_section(args, card, middle) / sm[card]
        print('card, scale is '+card+str(scales[card]))
        diff = scales[card] - args.scale
        print('diff is '+str(diff))
        print(" (np.abs(diff) > args.threshold)")
        print( (np.abs(diff) > args.threshold))
        print("(scales[card] > args.scale)")
        print((scales[card] > args.scale))
        print('scale is ')
        print(str(args.scale))
        if (np.abs(diff) > args.threshold) and (scales[card] > args.scale):
            print('breaking now')
            break
    scale = max(scales.values())
    print('scales are '+str(scales))
    if np.abs(scale - args.scale) < args.threshold:
        return middle
    elif go_left(scale):
        return bisect(args, sm, low, middle, go_left)
    else:
        return bisect(args, sm, middle, high, go_left)

def within_bounds(args, sm, endpoint):
    scales = {}
    for card in args.constraints:
        scales[card] = get_cross_section(args, card, endpoint) / sm[card]
    if max(scales.values()) < args.scale:
        return False
    return True

