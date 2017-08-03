from __future__ import print_function
import argparse
import json
import multiprocessing
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


def mp_call(arg):
    # pool.map supports only one iterable
    fct, args, kwargs = arg
    try:
        print('calling {} with args {}'.format(str(fct), str(args)))
        return fct(*args, **kwargs)
    except Exception as e:
        raise Exception('method {0} failed with "{1}", using args {2}, {3}'.format(fct, e, args, kwargs))

def mp_map(args):
    start = time.time()
    pool = multiprocessing.Pool(processes=len(args), maxtasksperchild=1)
    res = pool.map(mp_call, args)
    pool.close()
    pool.join()

    print('{} took {:.2f} seconds'.format(args[0][0], time.time() - start))
    return np.array(res)

def get_cross_section(args, card, value):
    with tempdir.TempDir() as sandbox:
        start = os.getcwd()
        os.chdir(sandbox)
        os.makedirs('source')
        os.makedirs('madgraph')
        subprocess.call(['tar', 'xaf', args.gridpack, '--directory=source'])

        with open(os.path.join('source', args.inparams)) as f:
            inparams = f.readlines()

        with tarfile.open(os.path.join(start, args.madgraph), "r:gz") as f:
            f.extractall('madgraph')

        with open(os.path.join('madgraph', args.outparams)) as f:
            outparams = f.readlines()

        with open(os.path.join('madgraph', args.outparams), 'w') as f:
            pattern = re.compile('(\d*) ([\de\+\-\.]*) (#.*) ')
            coefficients = []
            for operator in operators:
                # FIXME broken for more than one coefficient
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

        subprocess.check_output(['python', os.path.join('madgraph', 'bin', 'mg5_aMC'), '-f', os.path.join(start, card)])

        shutil.copy(os.path.join('source', 'process/madevent/Cards/run_card.dat'), 'processtmp/Cards')
        shutil.copy(os.path.join('source', 'process/madevent/Cards/grid_card.dat'), 'processtmp/Cards')
        shutil.copy(os.path.join('source', 'process/madevent/Cards/me5_configuration.txt'), 'processtmp/Cards')
        with open('processtmp/Cards/me5_configuration.txt', 'a') as f:
            print('run_mode = 2', file=f)
            print('nb_core = {0}'.format(args.cores), file=f)
            print('lhapdf = /cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/lhapdf/6.1.6/share/LHAPDF/../../bin/lhapdf-config', file=f)
            print('automatic_html_opening = False', file=f)


        with open('processtmp/Cards/run_card.dat', 'a') as f:
            print(' {} =  nevents'.format(args.events), file=f)
            print('.false. =  gridpack', file=f)

        output = subprocess.check_output(['./processtmp/bin/generate_events', '-f'])
        m = re.search("Cross-section :\s*(.*) \+", output)

        subprocess.call(['tar cJpsf {} processtmp/'.format(os.path.join(start, 'diagrams.tar.xz'))], shell=True)

        os.chdir(start)
        print('getting cross section for {} with {}={}'.format(card, args.coefficient, str(value)))
        print('returning '+str(float(m.group(1)) if m else None))
        return float(m.group(1)) if m else np.nan


# def bisect(args, sm, low, high, move_left):
#     print('calling bisect with sm {} low {} high {}'.format(str(sm), str(low), str(high)))
#     middle = (low + high) / 2.
    # scales = mp_map([(get_cross_section, [args, c, middle], {}) for c in args.constraints]) / sm

#     if np.abs(scales.max() - args.scale) < args.threshold:
#         return middle
#     elif move_left(scales.max()):
#         return bisect(args, sm, low, middle, move_left)
#     else:
#         return bisect(args, sm, middle, high, move_left)

def map_scales(args, sm, values):
    res = mp_map([(get_cross_section, [args, c, v], {}) for v in values for c in args.constraints])
    res = res.reshape(len(values), len(args.constraints))

    print('values are {}'.format(str(values)))
    print('inputs are {}'.format(str([[c, v] for v in values for c in args.constraints])))

    return [max(res[i] / sm) for i in range(len(values))]

def bisect(args, sm, left, right, left_to_right=True):
    middle = (left + right) / 2.
    y_left, y_right, y_middle = map_scales(args, sm, [left, right, middle])

    while True:
        print('starting bisect with left={} right={} left_to_right={}'.format(left, right, str(left_to_right)))
        if (y_left is None) or (y_right is None) or (y_middle is None):
            print('MadGraph failed to calculate cross section')
            return None

        if (y_left > args.scale) and (y_right > args.scale) and (y_middle > args.scale):
            print('no solution between {:.2f} and {:.2f}'.format(left, right))
            return None
        if (y_left < args.scale) and (y_right < args.scale) and (y_middle < args.scale):
            print('no solution between {:.2f} and {:.2f}'.format(left, right))
            return None

        if np.abs(y_middle - args.scale) < args.threshold:
            return middle

        if (left_to_right and y_middle > args.scale) or \
                (not left_to_right and y_middle < args.scale):
            left = middle
            y_left = y_middle
        if (left_to_right and y_middle < args.scale) or \
                (not left_to_right and y_middle > args.scale):
            right = middle
            y_right = y_middle
        middle = (left + right) / 2.
        y_middle = map_scales(args, sm, [middle])[0]
