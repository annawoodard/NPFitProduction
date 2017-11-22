from __future__ import print_function
from collections import defaultdict
import itertools
import os
import re
import shutil
import subprocess

import numpy as np

from NPFitProduction.NPFitProduction.utils import cartesian_product, TupleKeyDict, TempDir


class CrossSectionScan(object):
    """A container for cross section scans over Wilson coefficient values.

    """

    def __init__(self, fn=None):
        self.points = TupleKeyDict()
        self.cross_sections = TupleKeyDict()
        self.scales = TupleKeyDict()
        self.fit_constants = TupleKeyDict()
        self.fit_errs = TupleKeyDict()

        if fn is not None:
            if isinstance(fn, list):
                self.loadmany(fn)
            elif os.path.isfile(fn):
                self.load(fn)

    def load(self, fn):
        info = np.load(fn)
        self.points = TupleKeyDict(info['points'][()])
        self.cross_sections = TupleKeyDict(info['cross_sections'][()])
        self.scales = TupleKeyDict(info['scales'][()])
        self.fit_constants = TupleKeyDict(info['fit_constants'][()])
        self.fit_errs = TupleKeyDict(info['fit_errs'][()])

    def loadmany(self, files):
        for f in files:
            state = (self.points, self.cross_sections)
            try:
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
            except Exception as e:
                self.points = state[0]
                self.cross_sections = state[1]
                print('skipping bad input file {}: {}'.format(f, e))

    def add(self, points, cross_section, process, coefficients):
        argsort = sorted(range(len(coefficients)), key=lambda k: coefficients[k])
        points = points[:, argsort]
        coefficients = tuple(sorted(coefficients))
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

    def deduplicate(self, coefficients, process):
        """Deduplicate points

        Calculate the average cross section for each duplicated point,
        then merge the duplicated points into a single point with cross
        section equal to the average. Does not currently handle points
        calculated with different numbers of events (in which case the
        average should be weighted). Note this should only be used when
        producing samples, where points could be duplicated for
        efficient parallelization. For Madgraph-only scans, calculation
        of the cross section is fast enough that you should avoid
        requesting duplicated points rather than doing so and
        de-duplicating afterwards.

        See https://stackoverflow.com/questions/31878240
        """
        sort = np.lexsort(self.points[coefficients][process].T)
        mask = np.append(True, np.any(np.diff(self.points[coefficients][process][sort], axis=0), axis=1))
        tag = mask.cumsum() - 1
        averages = np.bincount(tag, self.cross_sections[coefficients][process][sort]) / np.bincount(tag)
        self.cross_sections[coefficients][process] = averages
        self.points[coefficients][process] = self.points[coefficients][process][sort][mask]

    def update_scales(self, coefficients, process):
        sm = np.array([0] * len(coefficients))  # SM point has all coefficients set to 0
        points = self.points[coefficients][process]
        cross_sections = self.cross_sections[coefficients][process]
        sm_indices = np.where((points == sm).all(axis=1))[0]
        if len(sm_indices) == 0:
            raise RuntimeError('scan does not contain the SM point for coefficients {} and process {}'.format(coefficients, process))
        sm_cross_section = np.mean(cross_sections[sm_indices])
        self.scales[coefficients][process] = (cross_sections / sm_cross_section)
        self.cross_sections['sm'][process] = sm_cross_section

    def prune(self, process, coefficients):
        self.points[coefficients] = dict((k, v) for k, v in self.points[coefficients].items() if k is not process)
        self.cross_sections[coefficients] = dict((k, v) for k, v in self.cross_sections[coefficients].items() if k is not process)

    def dump(self, filename):
        np.savez(
            filename,
            points=dict(self.points),
            cross_sections=dict(self.cross_sections),
            scales=dict(self.scales),
            fit_constants=dict(self.fit_constants),
            fit_errs=dict(self.fit_errs)
        )

    def model(self, points):
        rows, dim = points.shape
        pairs = list(itertools.combinations(range(0, dim), 2))

        constant = np.array([[1.0]] * rows)
        linear = points
        quad = points * points
        mixed = points[:, [x0 for x0, x1 in pairs]] * points[:, [x1 for x0, x1 in pairs]]

        return np.hstack([constant, linear, quad, mixed])

    def fit(self, maxpoints=None):
        """Perform a fit to describe how processes are scaled as a function of Wilson coefficients

        The matrix element M can be expressed in terms of the SM piece M_0 and NP
        pieces M_1, M_2, etc.

        For one Wilson coefficient c_1:
        M = M_0 + c_1 M_1
        sigma(c_1) ~ |M|^2 ~ s_0 + s_1 c_1 + s_2 c_1^2

        For two Wilson coefficients c_1, c_2:
        M = M_0 + c_1 M_1 + c_2 M_2
        sigma(c_1, c_2) ~ |M^2| ~ s_0 + s_1 c_1 + s_2 c_2 + s_3 c_1^2 + s_4 c_2^2 + s_5 c_1 c_2

        And similarly for more Wilson coefficients. For one operator, s_0, s_1, and s_2 can
        be solved for with three calculated points; for two operators, six points are needed, etc.
        In general 1 + 2 * d + (d - 1) * d / 2 points are needed, where d is the number of included
        operators. In practice, overconstraining the fit with more than the minimum number of points
        is helpful, because the MG calculation has associated errors.

        """
        for coefficients in self.points:
            for process, points in self.points[coefficients].items():
                if process not in self.scales[coefficients]:
                    self.update_scales(coefficients, process)
                indices = np.arange(0, len(points))
                np.random.shuffle(indices)
                train = indices
                if maxpoints is not None and maxpoints < len(indices):
                    train = indices[:maxpoints]
                print('found {} points for {}'.format(str(len(train)), process))
                matrix = self.model(points[train])
                scales = self.scales[coefficients][process]
                # the fit must go through the SM point, so we weight it
                weights = np.diag([100000 if (x[0] == 1. and np.all(x[1:]) == 0.) else 1 for x in matrix])
                self.fit_constants[coefficients][process], _, _, _ = np.linalg.lstsq(np.dot(weights, matrix), np.dot(scales[train], weights))
                if maxpoints is not None and maxpoints < len(indices):
                    test = indices[maxpoints:]
                    predicted = self.evaluate(coefficients, points[test], process)
                    self.fit_errs[coefficients][process] = (scales[test] - predicted) / scales[test] * 100

    def evaluate(self, coefficients, points, process):
        if isinstance(coefficients, basestring):
            coefficients = tuple([coefficients])
        if not self.fit_constants[coefficients]:
            self.fit()
        matrix = self.model(points)

        return np.dot(matrix, self.fit_constants[coefficients][process])


def get_maxes(scales, grid, coefficients):
    maxes = [scales[grid[:, i] > 0].max() for i in range(len(coefficients))]
    maxes += [scales[grid[:, i] < 0].max() for i in range(len(coefficients))]
    return maxes


def get_points(coefficients, coarse_scan, scale, interpolate_numvalues, calculate_numvalues, step=0.2, min_value=1e-11):
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
        step : float
            Change the range this much per iteration while searching for the desired range.

    """
    values = []
    if coefficients not in coarse_scan.points:
        raise RuntimeError('coarse scan is missing {}'.format(coefficients))
    coarse_scan.fit(coefficients)
    ranges = None
    # import pdb
    # pdb.set_trace()
    for process, points in coarse_scan.points[coefficients].items():
        values = [np.linspace(-1 * np.abs(points[:, i]).max(), np.abs(points[:, i]).max(), interpolate_numvalues) for i in range(len(coefficients))]
        grid = cartesian_product(*values)
        scales = coarse_scan.evaluate(coefficients, grid, process)
        while np.any([s < scale for s in get_maxes(scales, grid, coefficients)]):
            # case 1: process is not affected by operators, try to quickly reach the endpoint
            if (np.abs(grid).max() > (4 * np.pi) ** 2):  # convergence of the loop expansion requires c < (4 * pi)^2, see section 7 https://arxiv.org/pdf/1205.4231.pdf
                break
            values = [np.linspace(-1 * np.abs(grid[:, i]).max() * 2., np.abs(grid[:, i]).max() * 2., interpolate_numvalues) for i in range(len(coefficients))]
            grid = cartesian_product(*values)
            scales = coarse_scan.evaluate(coefficients, grid, process)
        while np.any([s > scale for s in get_maxes(scales, grid, coefficients)]):
            if (np.abs(grid).max()) < min_value:
                raise RuntimeError('fit did not converge')
            # case 2: we are above the endpoint, try to quickly zoom in
            values = [np.linspace(-1 * np.abs(grid[:, i]).max() / 2., np.abs(grid[:, i]).max() / 2., interpolate_numvalues) for i in range(len(coefficients))]
            grid = cartesian_product(*values)
            scales = coarse_scan.evaluate(coefficients, grid, process)
        while np.any([s < scale for s in get_maxes(scales, grid, coefficients)]):
            # we overshot, now slowly zoom out
            if (np.abs(grid).max() > (4 * np.pi) ** 2):
                break
            values = [np.linspace(-1 * np.abs(grid[:, i]).max() * (1. + step), np.abs(grid[:, i]).max() * (1. + step), interpolate_numvalues) for i in range(len(coefficients))]
            grid = cartesian_product(*values)
            scales = coarse_scan.evaluate(coefficients, grid, process)
        passed = grid[scales <= scale]
        endpoint = np.array([np.abs(passed[:, i]).max() for i in range(len(coefficients))])
        if ranges is None:
            ranges = endpoint
        else:
            ranges = np.amin(np.vstack([endpoint, ranges]), axis=0)
        print('ranges '+str(ranges))
        print('ranges shape '+str(ranges.shape))

    calculate_values = [np.hstack([np.zeros(1), np.linspace(-1. * ranges[i], ranges[i], calculate_numvalues - 1)]) for i in range(len(coefficients))]
    print('calcualte values '+str(calculate_values))
    grid = cartesian_product(*calculate_values)
    scales = coarse_scan.evaluate(coefficients, grid, process)

    return cartesian_product(*calculate_values)


def setup_model(base, madgraph, np_model, np_param_path, coefficients, process_card, cores, events, cards, point):
    """
    Setup the NP model and update the coefficient value

    Parameters
    ----------
        base : str
            The base directory where the tarballs are located.
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

    Returns
    ----------
        outdir : str
            The output madgraph processing directory.
    """

    subprocess.call(['tar', 'xaf', os.path.join(base, madgraph)])
    subprocess.call(['tar', 'xaf', os.path.join(base, np_model), '--directory=models'])

    if not np_param_path.startswith('models'):
        np_param_path = os.path.join('models', np_param_path)
    with open(np_param_path) as f:
        np_params = f.readlines()

    with open(np_param_path, 'w') as f:
        for line in np_params:
            for coefficient, value in zip(coefficients, point):
                line = re.sub('\d*.\d00000.*\# {0} '.format(coefficient), '{0} # {1}'.format(value, coefficient), line)

            f.write(line)

    subprocess.check_output(['python', os.path.join('bin', 'mg5_aMC'), '-f', os.path.join(base, process_card)])

    with open(os.path.join(base, process_card)) as f:
        card = f.read()
    outdir = re.search('\noutput (\S*)', card).group(1)
    carddir = os.path.join(outdir, 'Cards')

    shutil.copy(os.path.join(base, cards, 'run_card.dat'), carddir)
    shutil.copy(os.path.join(base, cards, 'grid_card.dat'), carddir)
    shutil.copy(os.path.join(base, cards, 'me5_configuration.txt'), carddir)
    with open(os.path.join(carddir, 'me5_configuration.txt'), 'a') as f:
        print('nb_core = {0}'.format(cores), file=f)

    with open(os.path.join(carddir, 'run_card.dat'), 'a') as f:
        print(' {} =  nevents'.format(events), file=f)

    return outdir


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
    with TempDir() as sandbox:
        os.chdir(sandbox)

        outdir = setup_model(start, madgraph, np_model, np_param_path, coefficients, process_card, cores, events, cards, point)

        output = subprocess.check_output(['./{}/bin/generate_events'.format(outdir), '-f'])
        m = re.search("Cross-section :\s*(.*) \+", output)
        os.chdir(start)

        try:
            return float(m.group(1))
        except (TypeError, AttributeError):
            raise RuntimeError('mg calculation failed')
