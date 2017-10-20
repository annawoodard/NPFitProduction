from __future__ import print_function
from collections import defaultdict
import argparse
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempdir
import time

import numpy as np

from EffectiveTTVProduction.EffectiveTTVProduction.utils import cartesian_product

class CrossSectionScan(object):
    """A container for cross section scans over Wilson coefficient values.

    """
    def __init__(self, files=None):
        self.points = defaultdict(dict)
        self.cross_sections = defaultdict(dict)
        self.signal_strengths = defaultdict(lambda: defaultdict(dict))

        if files:
            self.load(files)

    def load(self, files):
        for f in files:
            info = np.load(f)
            points = info['points'][()]
            cross_sections = info['cross_sections'][()]
            for coefficients in points:
                for process in points[coefficients]:
                    self.add(
                        points[coefficients][process],
                        cross_sections[coefficients][process],
                        process,
                        coefficients
                    )

    def add(self, points, cross_section, process, coefficients):
        coefficients = tuple(coefficients)
        if not isinstance(cross_section, np.ndarray):
            cross_section = np.array([cross_section])
        if len(points.shape) < 2:
            points = points.reshape(1, len(coefficients))
        if coefficients in self.points:
            if process in self.points[coefficients]:
                self.points[coefficients][process] = np.vstack([self.points[coefficients][process], points])
                self.cross_sections[coefficients][process] = np.hstack(
                    [self.cross_sections[coefficients][process], cross_section])
            else:
                self.points[coefficients][process] = points
                self.cross_sections[coefficients][process] = cross_section
        else:
            self.points[coefficients] = {process: points}
            self.cross_sections[coefficients] = {process: cross_section}

        self.update_signal_strengths(process, coefficients)

    def update_signal_strengths(self, process, coefficients):
        sm = np.array([0] * len(coefficients))  # SM point has all coefficients set to 0
        points = self.points[coefficients][process]
        cross_sections = self.cross_sections[coefficients][process]
        sm_indices = np.where((points == sm).all(axis=1))[0]
        sm_cross_section = np.mean(cross_sections[sm_indices])
        self.signal_strengths[coefficients][process] = (cross_sections / sm_cross_section)

    def prune(self, processes):
        for coefficients, points in self.points.items():
            self.points[coefficients] = dict((k, v) for k, v in points.items() if k in processes)
        for coefficients, cross_sections in self.cross_sections.items():
            self.cross_sections[coefficients] = dict((k, v) for k, v in cross_sections.items() if k in processes)

    def dump(self, filename):
        np.savez(
            filename,
            points=self.points,
            cross_sections=self.cross_sections
        )


def get_points(coefficients, coarse_scan, scale, numvalues):
    """Return a grid of points with dimensionality
    equal to the number of coefficients, and each axis spanning the
    minimum and maximum c_j for which NP / SM < scale, for any of the
    processes in the coarse scan.

    Parameters
    ----------
        coefficients : tuple of str
            The coefficients to be sampled.
        coarse_scan : CrossSectionScan
            The coarse scan to use for setting the coefficient value ranges.
        scale : float
            The maximum ratio of the (cross section)_NP / (cross section)_SM.
        numvalues : int
            The number of values to sample per coefficient.

    """
    values = []
    for i in range(len(coefficients)):
        low = None
        high = None
        try:
            for p, points in coarse_scan.points[coefficients].items():
                coarse_points = points[:, i]
                signal_strengths = coarse_scan.signal_strengths[coefficients][p]
                passed = coarse_points[signal_strengths < scale]
                if len(passed) < 2 or np.all(passed == 0):
                    # the scan is too coarse, so pick the smallest range containing 0
                    bottom = coarse_points[coarse_points < 0].max()
                    top = coarse_points[coarse_points > 0].min()
                    passed = np.array([bottom, top])
                if low is None:
                    low = min(passed)
                    high = max(passed)
                else:
                    low = max(low, min(passed))
                    high = min(high, max(passed))
            values += [np.hstack([np.zeros(1), np.linspace(low, high, numvalues)])]  # we add c_j = 0 to ensure we get the SM value
        except KeyError:
            raise ValueError('input scan missing {}'.format(','.join(coefficients)))

    return cartesian_product(*values)

def get_cross_section(madgraph, np_model, np_param_path, coefficients, process_card, cores, events, cards, point):
    """
    Update the Wilson coefficient value, run Madgraph, and return the calculated
    cross section.

    Parameters
    ----------
        madgraph : str
            Tarball containing madgraph.
        np_model : str
            Tarball containing NP model
        np_param_path : str
            Path (relative to the unpacked madgraph tarball) to the NP parameter card.
        coefficients : tuple of str
            Coefficients to scan.
        cores : int
            Number of cores to use.
        events : int
            Number of events to use for cross section calculation.
        cards : str
            Path to the cards directory (must contain run_card.dat, grid_card.dat, me5_configuration.txt
            and the parameter card pointed to by np_param_path).
        point : np.ndarray
            The values to set the coefficients to.
    """
    start = os.getcwd()
    with tempdir.TempDir() as sandbox:
        os.chdir(sandbox)

        subprocess.call(['tar', 'xaf', os.path.join(start, madgraph)])
        subprocess.call(['tar', 'xaf', os.path.join(start, np_model), '--directory=models'])

        if not np_param_path.startswith('models'):
            np_param_path = os.path.join('models', np_param_path)
        with open(np_param_path) as f:
            np_params = f.readlines()

        with open(np_param_path, 'w') as f:
            for line in np_params:
                for coefficient, value in zip(coefficients, point):
                    line = re.sub('\d*.\d00000.*\# {0} '.format(coefficient), '{0} # {1}'.format(value, coefficient), line)

                f.write(line)

        subprocess.check_output(['python', os.path.join('bin', 'mg5_aMC'), '-f', os.path.join(start, process_card)])

        with open(os.path.join(start, process_card)) as f:
            card = f.read()
        outdir = re.search('\noutput (\S*)', card).group(1)
        carddir = os.path.join(outdir, 'Cards')

        shutil.copy(os.path.join(start, cards, 'run_card.dat'), carddir)
        shutil.copy(os.path.join(start, cards, 'grid_card.dat'), carddir)
        shutil.copy(os.path.join(start, cards, 'me5_configuration.txt'), carddir)
        with open(os.path.join(carddir, 'me5_configuration.txt'), 'a') as f:
            print('nb_core = {0}'.format(cores), file=f)

        with open(os.path.join(carddir, 'run_card.dat'), 'a') as f:
            print(' {} =  nevents'.format(events), file=f)

        output = subprocess.check_output(['./{}/bin/generate_events'.format(outdir), '-f'])
        m = re.search("Cross-section :\s*(.*) \+", output)
        os.chdir(start)

        res = float(m.group(1)) if m else np.nan
        print('point {}\nresult {}'.format(str(point), str(res)))

        return res
