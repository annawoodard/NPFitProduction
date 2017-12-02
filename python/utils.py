from __future__ import print_function
import collections
import itertools
import numpy as np
import os
import re
import shutil
import subprocess
import tempfile


class TupleKeyDict(collections.MutableMapping):
    """Dictionary class for which keys are always tuples
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        if self.__keytransform__(key) not in self.store:
            self.store[self.__keytransform__(key)] = dict()
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        if isinstance(key, basestring):
            key = tuple([key])
        elif isinstance(key, list):
            key = tuple(key)
        return key


class TempDir(object):
    """ Class for temporary directories

        Creates a (named) directory which is deleted after use.
        All files created within the directory are destroyed.
        Might not work on windows when the files are still opened.

        Borrowed from module `tempdir` to avoid dependency issues.
    """

    def __init__(self, suffix="", prefix="tmp", basedir=None):
        self.name = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=basedir)

    def __del__(self):
        try:
            if self.name:
                self.dissolve()
        except AttributeError:
            pass

    def __enter__(self):
        return self.name

    def __exit__(self, *errstuff):
        self.dissolve()

    def dissolve(self):
        """remove all files and directories created within the tempdir"""
        if self.name:
            shutil.rmtree(self.name)
        self.name = ""

    def __str__(self):
        if self.name:
            return "temporary directory at: %s" % (self.name,)
        else:
            return "dissolved temporary directory"


def cartesian_product(*arrays):
    # https://stackoverflow.com/questions/11144513
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def sorted_combos(items, dimension):
    return [tuple(sorted(items)) for items in itertools.combinations(items, dimension)]

def clone_cards(
        sm_gridpack,
        np_model,
        sm_param_path,
        np_param_path,
        sm_card_path,
        outdir,
        lhapdf,
        extras=[]):
    """
    Clone the parameter card of a SM gridpack. All parameters from the SM
    card which exist in the NP card will be copied.

    Parameters
    ----------
        sm_gridpack : str
            SM gridpack to clone
        model : str
            Tarball containing NP model
        sm_param_path : str
            Path (relative to the unpacked SM gridpack) to the SM
            parameter card
        np_param_path : str
            Path (relative to the unpacked model tarball) to
            the NP parameter card.
        sm_card_path : str
            Path (relative to the unpacked SM gridpack) to the Cards
            directory
        outdir : str
            Directory to write cloned cards to
        lhapdf : str
            Path to lhapdf config to use
        extras : list
            Anything extra to save from the sm_gridpack
    """

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    with TempDir() as sandbox:
        os.makedirs('{}/sm'.format(sandbox))
        os.makedirs('{}/np'.format(sandbox))
        subprocess.call(['tar', 'xaf', sm_gridpack, '--directory={}/sm'.format(sandbox)])
        subprocess.call(['tar', 'xaf', np_model, '--directory={}/np'.format(sandbox)])

        with open(os.path.join(sandbox, 'sm', sm_param_path)) as f:
            sm_params = f.readlines()

        with open(os.path.join(sandbox, 'np', np_param_path)) as f:
            np_params = f.readlines()

        with open(os.path.join(outdir, os.path.split(np_param_path)[-1]), 'w') as f:
            pattern = re.compile('(\d*) ([\de\+\-\.]*) (#.*) ')
            for np_line in np_params:
                match = re.search(pattern, np_line)
                if match:
                    _, sm_value, sm_label = match.groups()
                    for sm_line in sm_params:
                        match = re.search(pattern, sm_line)
                        if match:
                            _, np_value, np_label = match.groups()
                            if np_label == sm_label:
                                np_line = re.sub(re.escape(sm_value), np_value, np_line)

                f.write(np_line)

        shutil.copy(os.path.join(sandbox, 'sm', sm_card_path, 'run_card.dat'), outdir)
        shutil.copy(os.path.join(sandbox, 'sm', sm_card_path, 'grid_card.dat'), outdir)
        shutil.copy(os.path.join(sandbox, 'sm', sm_card_path, 'me5_configuration.txt'), outdir)

        with open(os.path.join(outdir, 'me5_configuration.txt'), 'a') as f:
            print('run_mode = 2', file=f)
            print('lhapdf = {}'.format(lhapdf), file=f)
            print('automatic_html_opening = False', file=f)

        with open(os.path.join(outdir, 'run_card.dat'), 'a') as f:
            print('.false. =  gridpack', file=f)

        for entry in extras:
            shutil.copy(os.path.join(sandbox, 'sm', entry), outdir)
