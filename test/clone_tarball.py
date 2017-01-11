from __future__ import print_function
import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempdir
import time

import numpy as np

from EffectiveTTVProduction.EffectiveTTVProduction.operators import operators

parser = argparse.ArgumentParser(description='add Higgs Effective Lagrangian model to gridpack')
parser.add_argument('process', help='process card to run')
parser.add_argument('gridpack', help='gridpack to clone')
parser.add_argument('madgraph', help='madgraph directory with model')
parser.add_argument('inparams', help='param card to clone')
parser.add_argument('outparams', help='param card to update')
parser.add_argument('cores', help='number of cores to use')
parser.add_argument('operators', nargs='+', help='operators to scan')
parser.add_argument('points', help='json file with parameter points')
parser.add_argument('point', type=int, help='parameter space point')
args = parser.parse_args()

with tempdir.TempDir() as source:
    subprocess.call(['tar', 'xaf', args.gridpack, '--directory={0}'.format(source)])

    points = np.load(args.points)[()]

    with open(os.path.join(source, args.inparams)) as f:
    # with open('/afs/crc.nd.edu/user/a/awoodard/releases/effective-ttV-production/CMSSW_7_4_7/src/EffectiveTTVProduction/EffectiveTTVProduction/test/param_card.dat') as f:
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
                if operator in args.operators:
                    coefficients.append(points[operator][args.point])
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

        print('source is '+source)
        print('madgraph is '+madgraph)
        print('writing to '+os.path.join(madgraph, args.outparams))
        # import time
        # time.sleep(500000)
        subprocess.call(['python', os.path.join(madgraph, 'bin', 'mg5_aMC'), '-f', args.process])

        shutil.copy(os.path.join(source, 'process/madevent/Cards/run_card.dat'), 'processtmp/Cards')
        # shutil.copy('/afs/crc.nd.edu/user/a/awoodard/releases/effective-ttV-production/CMSSW_7_4_7/src/EffectiveTTVProduction/EffectiveTTVProduction/test/run_card.v24.dat', 'processtmp/Cards/run_card.dat')
        shutil.copy(os.path.join(source, 'process/madevent/Cards/grid_card.dat'), 'processtmp/Cards')
        shutil.copy(os.path.join(source, 'process/madevent/Cards/me5_configuration.txt'), 'processtmp/Cards')
        with open('processtmp/Cards/me5_configuration.txt', 'a') as f:
            print('run_mode = 2', file=f)
            print('nb_core = {0}'.format(args.cores), file=f)
            print('lhapdf = /afs/crc.nd.edu/user/a/awoodard/local/bin/lhapdf-config', file=f)

        subprocess.call(['./processtmp/bin/generate_events', '-f'])
        subprocess.call(['tar', 'xzf', 'processtmp/run_01_gridpack.tar.gz'])

        os.mkdir('process')
        shutil.move('madevent', 'process')
        shutil.move('run.sh', 'process')
        shutil.move(os.path.join(source, 'mgbasedir'), '.')
        shutil.move(os.path.join(source, 'runcmsgrid.sh'), '.')

        with open('point.json', 'w') as f:
            json.dump({
                'operators': operators,
                'coefficients': coefficients,
                'point': args.point,
                'process': args.process.split('/')[-1].replace('.dat', '')
            }, f)

        subprocess.call(['tar', 'cJpsf', 'gridpack.tar.xz', 'mgbasedir', 'process', 'runcmsgrid.sh', 'point.json'])
        subprocess.call(['tar cJpsf diagrams.tar.xz processtmp/SubProcesses/'], shell=True)

