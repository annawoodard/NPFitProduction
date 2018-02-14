from __future__ import print_function
import itertools
import logging
import os
import re
import shutil
import subprocess
import tabulate
import time
import glob
import xml.etree.ElementTree as ET

import numpy as np
import scipy
import scipy.optimize

from NPFitProduction.NPFitProduction.utils import cartesian_product, TupleKeyDict, TempDir, sorted_combos

logger = logging.getLogger(__name__)

# TODO keep track of dimensions
class CrossSectionScan(object):
    """A container for cross section scans over Wilson coefficient values.

    """

    def __init__(self, fn=None):
        self.points = TupleKeyDict()
        self.cross_sections = TupleKeyDict()
        # TODO better name
        self.errs = TupleKeyDict()
        self.fit_constants = TupleKeyDict()
        self.fit_constants_errs = TupleKeyDict()
        self.covariances = TupleKeyDict()

        if fn is not None:
            if isinstance(fn, list):
                self.loadmany(fn)
            elif os.path.isfile(fn):
                self.load(fn)
            else:
                raise IOError('cannot find {}'.format(fn))

    def __repr__(self):
        msg = ['{:40s} {:5s} {:5s} {:>50s} {:>55s} {:10s}'.format(
            'coefficients', 'process', 'points', 'min values', 'max values', 'max scaling')]
        for coefficients in self.points:
            for process, points in self.points[coefficients].items():
                try:
                    # scales = [1]
                    scales, _ = self.scales(coefficients, process)
                except RuntimeError:
                    scales = [-99.]
                msg += ['{:40s} {:5s} {:5d} {:>55s} {:>55s} {:.1f}'.format(
                    '({})'.format(', '.join(coefficients)),
                    process,
                    len(points),
                    '({})'.format(', '.join('{}'.format(self.round(i, 1)) for i in np.amin(points, axis=0))),
                    '({})'.format(', '.join('{}'.format(self.round(i, 1)) for i in np.amax(points, axis=0))),
                    max(scales)
                    )
                ]
        return '\n'.join(msg)

    def load(self, fn):
        try:
            info = np.load(fn)
        except (UnicodeError, UnicodeDecodeError):
            info = np.load(fn, encoding='utf8')
        info = np.load(fn)
        self.points = TupleKeyDict(info['points'][()])
        self.cross_sections = TupleKeyDict(info['cross_sections'][()])
        self.fit_constants = TupleKeyDict(info['fit_constants'][()])
        try:
            self.errs = TupleKeyDict(info['errs'][()])
        except:
            for coefficients in self.points:
                self.errs[coefficients] = {}
                for process in self.points[coefficients]:
                    self.errs[coefficients][process] = np.zeros(self.cross_sections[coefficients][process].shape)

    def loadmany(self, files):
        """Load a list of files
        """
        files = sum([glob.glob(f) for f in files], [])

        for f in files:
            state = (self.points, self.cross_sections, self.errs)
            try:
                try:
                    info = np.load(f)
                except (UnicodeError, UnicodeDecodeError):
                    info = np.load(f, encoding='utf8')
                points = info['points'][()]
                cross_sections = info['cross_sections'][()]
                try:
                    errs = info['errs'][()]
                except:
                    errs = TupleKeyDict()
                    for coefficients in points:
                        errs[coefficients] = {}
                        for process in points[coefficients]:
                            errs[coefficients][process] = np.zeros(cross_sections[coefficients][process].shape)
                for coefficients in points:
                    for process in points[coefficients]:
                        self.add(
                            points[coefficients][process],
                            cross_sections[coefficients][process],
                            errs[coefficients][process],
                            process,
                            coefficients
                        )
            except Exception as e:
                self.points = state[0]
                self.cross_sections = state[1]
                self.errs = state[2]
                print('skipping bad input file {}: {}'.format(f, e))

    def add(self, points, cross_section, err, process, coefficients):
        if isinstance(points, list):
            points = np.array(points)
        if len(points.shape) < 2:
            points = points.reshape(1, len(coefficients))
        argsort = sorted(range(len(coefficients)), key=lambda k: coefficients[k])
        points = points[:, argsort]
        coefficients = tuple(sorted(coefficients))
        if not isinstance(cross_section, np.ndarray):
            cross_section = np.array([cross_section])
        if not isinstance(err, np.ndarray):
            err = np.array([err])
        if coefficients in self.points:
            if process in self.points[coefficients]:
                self.points[coefficients][process] = np.vstack([self.points[coefficients][process], points])
                self.cross_sections[coefficients][process] = np.hstack(
                    [self.cross_sections[coefficients][process], cross_section])
                self.errs[coefficients][process] = np.hstack(
                    [self.errs[coefficients][process], err])
            else:
                self.points[coefficients][process] = points
                self.cross_sections[coefficients][process] = cross_section
                self.errs[coefficients][process] = err
        else:
            self.points[coefficients] = {process: points}
            self.cross_sections[coefficients] = {process: cross_section}
            self.errs[coefficients] = {process: err}

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

    def scales(self, coefficients, process):
        sm_cross_sections = []
        sm_errs = []
        for c in self.points:
            if process in self.points[c]:
                sm_indices = np.where(np.all(self.points[c][process] == 0, axis=1))[0]
                sm_cross_sections += self.cross_sections[c][process][sm_indices].tolist()
                sm_errs += self.errs[c][process][sm_indices].tolist()
        if len(sm_cross_sections) == 0:
            raise RuntimeError('scan does not contain the SM point for process {}'.format(process))

        sm_cross_section = np.mean(np.array(sm_cross_sections))
        sm_err = np.sqrt(sum(np.array(sm_errs) ** 2)) / len(sm_errs)

        self.cross_sections['sm'][process] = sm_cross_section
        self.errs['sm'][process] = sm_err

        cross_sections = self.cross_sections[coefficients][process]
        errs = self.errs[coefficients][process]

        scale = cross_sections / sm_cross_section
        scale_err = scale * np.sqrt((errs / cross_sections) ** 2 + (sm_err / sm_cross_section) ** 2)

        return scale, scale_err

    def prune(self, process, coefficients):
        self.points[coefficients] = dict((k, v) for k, v in self.points[coefficients].items() if k != process)
        self.cross_sections[coefficients] = dict((k, v) for k, v in self.cross_sections[coefficients].items() if k != process)

    def dump(self, filename):
        np.savez(
            filename,
            points=dict(self.points),
            cross_sections=dict(self.cross_sections),
            fit_constants=dict(self.fit_constants),
            errs=dict(self.errs)
        )

    def model(self, points):
        rows, dim = points.shape
        pairs = sorted(list(itertools.combinations(range(0, dim), 2)))

        constant = np.array([[1.0]] * rows)
        linear = points
        quad = points * points
        mixed = points[:, [x0 for x0, x1 in pairs]] * points[:, [x1 for x0, x1 in pairs]]

        return np.hstack([constant, linear, quad, mixed])

    def construct(self, process, coefficients):
        # from IPython.core import debugger
        # debugger.Pdb().set_trace()
        if isinstance(coefficients, str):
            coefficients = tuple([coefficients])
        pairs = sorted(list(itertools.combinations(range(0, len(coefficients)), 2)))

        if () not in self.fit_constants[process].keys():
            self.fit()
        res = self.fit_constants[process][()]

        for linear in coefficients:
            constant = self.fit_constants[process][(linear,)]
            res = np.concatenate([res, constant])
        for quad in coefficients:
            constant = self.fit_constants[process][(quad, quad)]
            res = np.concatenate([res, constant])
        for mixed in pairs:
            try:
                constant = self.fit_constants[process][(coefficients[mixed[0]], coefficients[mixed[1]])]
            except KeyError:
                constant = self.fit_constants[process][(coefficients[mixed[1]], coefficients[mixed[0]])]
            res = np.concatenate([res, constant])

        return res

    def dump_constants(self, process, coefficients, labels=None, sigfigs=10, tweak=1.):
        # from IPython.core import debugger
        # debugger.Pdb().set_trace()
        pairs = sorted(list(itertools.combinations(range(0, len(coefficients)), 2)))
        if labels is None:
            labels = dict((c, c) for c in coefficients)

        table = []
        # constants = self.construct(process, coefficients)
        row = ['1.0']
        def strip_zeros(num, sf):
            return '{:.15f}'.format(self.round(num, sf)).rstrip('0')
        def leading_zeros(num):
            left, right = '{:.15f}'.format(num).split('.')
            return len(right) - len(right.lstrip('0'))
        for i in coefficients:
            constant = self.fit_constants[process][(i,)][0]
            err = self.fit_constants_errs[process][(i,)][0] * tweak
            sf = max(leading_zeros(err) - leading_zeros(constant), 0)
            print(i, leading_zeros(err), leading_zeros(constant), sf)
            row += ['\\num{{{} +- {}}}'.format(strip_zeros(constant, sf), strip_zeros(err, 0))]
        table.append(row)
        for i in coefficients:
            row = [labels[i]]
            for j in coefficients:
                try:
                    constant = self.fit_constants[process][(i, j)][0]
                    err = self.fit_constants_errs[process][(i, j)][0] * tweak
                    sf = max(leading_zeros(err) - leading_zeros(constant), 0)
                    print(i, j, leading_zeros(err), leading_zeros(constant), sf)
                    row += ['\\num{{{} +- {}}}'.format(strip_zeros(constant, sf), strip_zeros(err, 0))]
                except KeyError:
                    row += ['-']
                    # constant = self.fit_constants[process][(j, i)]
            table.append(row)

        print(tabulate.tabulate(table, headers=[''] + [labels[i] for i in coefficients], tablefmt="latex_raw"))

    def convert(self, conversion):
        for coefficients in self.points:
            for process in self.points[coefficients]:
                for column, c in enumerate(coefficients):
                    self.points[coefficients][process][:, column] *= conversion[c]


    def fit(self, maxpoints=None, dimensions=None):
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
        self.fit_constants = TupleKeyDict()
        self.fit_constants_raw = TupleKeyDict()
        self.fit_constants_weights = TupleKeyDict()
        for coefficients in self.points:
            for process, points in self.points[coefficients].items():
                # TODO switch back to only one dimension fit
                if (dimensions is None) or (len(coefficients) in dimensions):
                    scales, variances = self.scales(coefficients, process)
                    indices = list(range(len(points)))
                    if (maxpoints is not None) and (maxpoints < len(points)):
                        sm_indices = np.where(np.all(points == 0, axis=1))[0]
                        np_indices = np.where(np.any(points != 0, axis=1))[0]
                        # be sure not to truncate the SM point
                        indices = sm_indices.tolist() + np_indices.tolist()
                        np.random.shuffle(indices)
                        points = points[indices[:maxpoints]]
                        scales = scales[indices[:maxpoints]]
                        variances = variances[indices[:maxpoints]]
                    print('fitting {} {} using {} points'.format(str(coefficients), process, str(len(points))))
                    A = self.model(points)
                    rows, cols = A.shape
                    # the fit must go through the SM point, so weight it
                    np.clip(variances, 1e-15, 1e15, variances)
                    # weights = np.diag([1 if row else 1 for row in np.all(points == 0, axis=1)])
                    # weights = np.diag([1e10 if row else 1 for row in np.all(points == 0, axis=1)])
                    # weights = np.diag(1 / (variances))
                    weights = np.diag(1 / (variances)) + np.diag([1e10 if row else 1 for row in np.all(points == 0, axis=1)])
                    # weights = np.diag(1 / (variances ** 2)) + np.diag([1e10 if row else 1 for row in np.all(points == 0, axis=1)])
                    # let the NP scaling per point be represented by B
                    # and the fit constants we want by X;
                    # this solves AX = B by computing the X which minimizes ||B - AX||^2
                    constants, _, _, _ = np.linalg.lstsq(np.dot(weights, A), np.dot(scales, weights))
                    # V_scales= np.diag(variances ** 2)
                    V_scales= np.diag(variances)
                    self.covariances[coefficients][process] = np.linalg.inv(np.dot(A.T, np.dot(np.linalg.inv(V_scales), A)))
                    constant_variances = np.diag(np.sqrt(np.abs(self.covariances[coefficients][process])))
                    pairs = sorted(list(itertools.combinations(range(0, len(coefficients)), 2)))
                    linear = [(l,) for l in coefficients]
                    quad = [(q, q) for q in coefficients]
                    mixed = [(coefficients[mixed[0]], coefficients[mixed[1]]) for mixed in pairs]
                    for i, term in enumerate([()] + linear + quad + mixed):
                        self.update(process, term, constants[i], constant_variances[i], len(points))

    def update(self, process, term, value, err, weight):
        if term in self.fit_constants[process]:
            self.fit_constants_weights[process][term] += [weight]
            self.fit_constants_raw[process][term] += [value]
            self.fit_constants[process][term] = np.array([np.average(
                self.fit_constants_raw[process][term],
                axis=0,
                weights=self.fit_constants_weights[process][term]
            )])
        else:
            self.fit_constants_weights[process][term] = [weight]
            self.fit_constants_raw[process][term] = [value]
            self.fit_constants[process][term] = np.array([value])
        self.fit_constants_errs[process][term] = np.array([err])

    def upscale(self, target_coefficients):
        for source_coefficients in self.points:
            subset = (set(source_coefficients) & set(target_coefficients)) == set(source_coefficients)
            if len(source_coefficients) < len(target_coefficients) and (subset is True):
                source_cols = dict((c, i) for i, c in enumerate(source_coefficients))
                for process in self.points[source_coefficients]:
                    source_rows = range(0, len(self.points[source_coefficients][process]))
                    res = np.zeros((len(source_rows), len(target_coefficients)))
                    for target_index, target_coefficient in enumerate(target_coefficients):
                        if target_coefficient in source_coefficients:
                            source_index = source_cols[target_coefficient]
                            res[:, target_index] = self.points[source_coefficients][process][source_rows][:, source_index]
                    self.add(res, self.cross_sections[source_coefficients][process][source_rows], process, target_coefficients)

    def evaluate(self, coefficients, points, process):
        if isinstance(coefficients, str):
            coefficients = tuple([coefficients])
        if isinstance(points, list):
            points = np.array(points)
        if len(points.shape) == 1:
            points = points.reshape((len(points), 1))

        A = self.model(points)
        constants = self.construct(process, coefficients)

        return np.dot(A, constants)

    def round(self, f, sig_figs):
        if isinstance(f, np.ndarray):
            res = f.copy()
            for i in range(len(res)):
                res[i] = float('{0:.{1}e}'.format(res[i], sig_figs))

            return res
        else:
            return float('{0:.{1}e}'.format(f, sig_figs))

    def test(self, coefficients, process):
        points = self.points[coefficients][process]
        A = self.model(points)
        constants = self.construct(process, coefficients)

        mg = self.scales(coefficients, process)
        for i in range(30):
            rounded = self.round(constants, i)
            fit = np.dot(A, rounded)
            err = (mg - fit) / mg * 100.
            print('{:5d} {:20.10f} {:20.10f}'.format(i, np.mean(err), max(err)))

    def test_errs(self, coefficients, process, tweak=1.):
        # http://people.duke.edu/~hpgavin/SystemID/CourseNotes/linear-least-squres.pdf
        points = self.points[coefficients][process]
        mg, _ = self.scales(coefficients, process)
        fit = self.evaluate(coefficients, points, process)
        A = self.model(points)
        Vy = np.dot(A, np.dot(self.covariances[coefficients][process] * tweak, A.T))
        spe = np.sqrt(np.diag(Vy))
        low = mg - 1.96 * spe
        high = mg + 1.96 * spe
        inside = len(fit[(fit > low) & (fit < high)]) / float(len(fit))

        abs_per_err = (np.abs((mg - fit) / mg)) * 100.
        print('average percent error {:.4f}, {:.4f}'.format(np.mean(abs_per_err), np.std(abs_per_err)))
        print('{:.1f}% inside 95% band'.format(inside * 100.))

    def dataframe(self, coefficients, evaluate_points=None):
        try:
            import pandas as pd
        except AttributeError as e:
            raise(e)
        processes = self.points[coefficients].keys()
        columns = list(coefficients) + processes
        if evaluate_points is None:
            df = pd.DataFrame(columns=list(coefficients) + processes)
            for process, points in self.points[coefficients].items():
                data = dict((coefficient, points[:, column]) for column, coefficient in enumerate(coefficients))
                data[process], _ = self.scales(coefficients, process)
                df = df.merge(pd.DataFrame(data), 'outer')
        else:
            df = pd.DataFrame(columns=list(coefficients) + ['{} fit'.format(p) for p in processes])
            for process in processes:
                scales = self.evaluate(coefficients, evaluate_points, process)
                data = dict((coefficient, evaluate_points[:, column]) for column, coefficient in enumerate(coefficients))
                data[process] = scales
                df = df.merge(pd.DataFrame(data), 'outer')

        return df


def get_edge_points(column, mins, maxes, edge, coefficients, num):
    values = []
    for j in range(len(coefficients)):
        if column != j:
            values += [np.linspace(mins[j], maxes[j], num)]
        else:
            values += [np.array([edge])]

    return cartesian_product(*values)

def get_perimeter(mins, maxes, coefficients, numvalues):
    res = None
    for i in range(len(coefficients)):
        if res is not None:
            res = np.vstack([
                res,
                get_edge_points(i, mins, maxes, maxes[i], coefficients, numvalues),
                get_edge_points(i, mins, maxes, mins[i], coefficients, numvalues)
            ])
        else:
            res = np.vstack([
                get_edge_points(i, mins, maxes, maxes[i], coefficients, numvalues),
                get_edge_points(i, mins, maxes, mins[i], coefficients, numvalues)
            ])

    return res

def get_bounds(coefficients, coarse_scan, scale):
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

    """

    # coarse_scan.fit()
    maxes = {}
    mins = {}

    start = time.time()
    for column, coefficient in enumerate(coefficients):
        coefficient_sets = [x for x in coarse_scan.points.keys() if coefficient in x]
        if len(coefficient_sets) == 0:
            raise RuntimeError('coarse scan is missing {}'.format(coefficient))
        coefficient_set = sorted(coefficient_sets, key=lambda x: len(x))[-1]
        for process in coarse_scan.points[coefficient_set]:
            s0, s1, s2 = coarse_scan.construct(process, coefficient)
            def f(point):
                return scale - (s0 + s1 * point + s2 * point * point)
            low = scipy.optimize.fsolve(f, [-4 * np.pi ** 2])[0]
            high = scipy.optimize.fsolve(f, [4 * np.pi ** 2])[0]

            if column in maxes:
                maxes[column] = min(maxes[column], high)
                mins[column] = max(mins[column], low)
            else:
                maxes[column] = high
                mins[column] = low

    # for process in coarse_scan.points[coefficient_set]:
    #     num_sampled_points = 1000000
    #     sampled_points = np.zeros((num_sampled_points, len(coefficients)))
    #     for i in range(num_sampled_points):
    #         point = []
    #         for column, coefficient in enumerate(coefficients):
    #             point += [np.random.uniform(mins[column], maxes[column])]
    #         sampled_points[i] = np.array([point])

    #     scales = coarse_scan.evaluate(coefficients, sampled_points, process).ravel()
    #     sampled_points = sampled_points[scales < scale]
    #     print('now len is ', process, coefficients, len(sampled_points))

    #     for column, low in enumerate(np.amin(sampled_points, axis=0)):
    #         mins[column] = max(mins[column], low)
    #     for column, high in enumerate(np.amax(sampled_points, axis=0)):
    #         maxes[column] = min(maxes[column], high)
    # print('got bounds for {} with {} points in {:.1f} seconds'.format(str(coefficients), len(sampled_points), time.time() - start))
    return mins, maxes

# def get_bounds(coefficients, coarse_scan, scale, interpolate_numvalues, step=0.2, min_value=1e-11):
#     """Return a grid of points with dimensionality
#     equal to the number of coefficients, and each axis spanning the
#     minimum and maximum c_j for which NP / SM < scale, for any of the
#     processes in the coarse scan.
#     Parameters
#     ----------
#         coefficients : tuple of str
#             The coefficients to be sampled.
#         coarse_scan : CrossSectionScan
#             The coarse scan to use for setting the coefficient value ranges.
#         scale : float
#             The maximum ratio of the (cross section)_NP / (cross section)_SM.
#         numvalues : int
#             The number of values to sample per coefficient.
#         step : float
#             Change the range this much per iteration while searching for the desired range.
#     """

#     coarse_scan.fit()
#     maxes = {}
#     mins = {}

#     start = time.time()
#     for column, coefficient in enumerate(coefficients):
#         if coefficient not in coarse_scan.points:
#             raise RuntimeError('coarse scan is missing {}'.format(coefficient))
#         for process, points in coarse_scan.points[coefficient].items():
#             points = np.array(np.linspace(points.min(), points.max(), interpolate_numvalues))
#             scales = coarse_scan.evaluate(coefficient, points, process)
#             # case 1: process is not affected by operators, try to quickly reach the endpoint
#             while scales[points > 0].max() < scale:
#                 if (np.abs(points).max() > (4 * np.pi) ** 2):  # convergence of the loop expansion requires c < (4 * pi)^2, see section 7 https://arxiv.org/pdf/1205.4231.pdf
#                     break
#                 points[points > 0] *= 2.
#                 scales = coarse_scan.evaluate(coefficient, points, process)
#             while scales[points < 0].max() < scale:
#                 if (np.abs(points).max() > (4 * np.pi) ** 2):
#                     break
#                 points[points < 0] *= 2.
#                 scales = coarse_scan.evaluate(coefficient, points, process)
#             # case 2: we are way above the target scale, try to quickly zoom in
#             while scales[points > 0].max() > scale:
#                 if (np.abs(points).max()) < min_value:
#                     raise RuntimeError('fit did not converge')
#                 points[points > 0] /= 2.
#                 scales = coarse_scan.evaluate(coefficient, points, process)
#             while scales[points < 0].max() > scale:
#                 if (np.abs(points).max()) < min_value:
#                     raise RuntimeError('fit did not converge')
#                 points[points < 0] /= 2.
#                 scales = coarse_scan.evaluate(coefficient, points, process)
#             # case 3: we overshot, now slowly zoom out
#             while scales[points > 0].max() < scale:
#                 if (np.abs(points).max() > (4 * np.pi) ** 2):
#                     break
#                 points[points > 0] *= (1. + step)
#                 scales = coarse_scan.evaluate(coefficient, points, process)
#             while scales[points < 0].max() < scale:
#                 if (np.abs(points).max() > (4 * np.pi) ** 2):
#                     break
#                 points[points < 0] *= (1. + step)
#                 scales = coarse_scan.evaluate(coefficient, points, process)
#             if column in maxes:
#                 maxes[column] = min(maxes[column], points.max())
#                 mins[column] = max(mins[column], points.min())
#             else:
#                 maxes[column] = points.max()
#                 mins[column] = points.min()

#     print('got bounds for {} in {:.1f} seconds'.format(str(coefficients), time.time() - start))
#     return mins, maxes

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


def setup_sandbox(madgraph, np_model, np_param_path, coefficients, process_card, cores, events, cards, point, sandbox):
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
    os.makedirs(sandbox)

    os.chdir(sandbox)

    outdir = setup_model(start, madgraph, np_model, np_param_path, coefficients, process_card, cores, events, cards, point)
    print('sandbox is ', sandbox)
    print('outdir is ', outdir)

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
        m = re.search("Cross-section :\s*(.*) \+\- (.*) pb", output)
        os.chdir(start)
        print(output)

        try:
            return float(m.group(1)), float(m.group(2))
        except (TypeError, AttributeError):
            raise RuntimeError('mg calculation failed')

def get_coefficient_ids(lines, coefficients):
    coefficient_ids = {}
    for line in lines:
        if 'Block' in line:
            current_block = line.split()[-1].strip()
        m = re.search('(\d*) [\d.e\-\+]* # (\S*)', line)
        if m:
            id, coef = m.groups()
            if coef in coefficients:
                model_block = current_block
                coefficient_ids[coef] = id
    return coefficient_ids, model_block

def write_reweight_card(param_card, reweight_card, numpoints, coefficients, mins, maxes):
    """Write a reweight card for Madgraph

    This should be good-to-go for multidimensions
    """
    with open(param_card, 'r') as f:
        coefficient_ids, model_block = get_coefficient_ids(f.readlines(), coefficients)
    print('ids ', coefficient_ids)

    with open(reweight_card, 'w') as f:
        for _ in range(numpoints):
            f.write('launch\n')
            for column, coef in enumerate(coefficients):
                coef_value = np.random.uniform(mins[column], maxes[column])
                f.write('set {} {} {:.6f}\n'.format(model_block, coefficient_ids[coef], coef_value))

def parse_lhe_weights(lhe, coefficients):
    """Parse weights from LHE file

    Returns
    -------
    numpy.ndarray
        Array of weights, where each row corresponds to an event
        and each column corresponds to a point
    numpy.ndarray
        Array of points, where each row corresponds to a point and
        each column corresponds to a coefficient
    """
    print('parsing {}'.format(lhe))

    tree = ET.parse(lhe)
    root = tree.getroot()
    points = []
    slha = list(root.iter('slha'))[0]
    coefficient_ids, model_block = get_coefficient_ids(slha.text.splitlines(), coefficients)
    num_points = len(list(root.iter('weight')))
    points = np.zeros((num_points, len(coefficients)))
    for i, weight in enumerate(root.iter('weight')):
        values = dict(re.findall('param_card {} (\d*) ([\d.e\-\+]*)'.format(model_block), weight.text))
        for j, c in enumerate(coefficients):
            try:
                points[i][j] = values[coefficient_ids[c]]
            except KeyError as e:
                print('skipping {} {}: {}'.format(j, c, e))
    start_values = np.zeros(len(coefficients))
    for id, value in re.findall('param_card {} (\d*) [\d.e\-\+]* # orig: ([\d.e\-\+]*)'.format(model_block), weight.text):
        for i, c in enumerate(coefficients):
            if id == coefficient_ids[c]:
                start_values[i] = value

    num_events = len(root[2:])
    print('will process {} points in {} events'.format(num_points, num_events))
    weights = np.zeros((num_events, num_points))
    for i, event in enumerate(root[2:]):
        weights[i] = np.array([float(wgt.text) for wgt in event.iter('wgt')])
        if i % 10000 == 0:
            print('completed {} / {} events'.format(i, num_events))

    return start_values, points, weights

