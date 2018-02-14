import glob
import imp
import os

import numpy as np

from lobster import cmssw
from lobster.core import *

# email = 'changeme@changeme.edu'  # uncomment to have notification sent here when processing completes
dimension = 8  # number of coefficients to change per scan
version = 'ttV/cross_sections/30/{}d'.format(dimension)  # you should increment this each time you make changes
# coefficients = ['c2B', 'c2G', 'c2W', 'c3G', 'c3W', 'c6', 'cA', 'cB', 'cG', 'cH', 'cHB', 'cHL', 'cHQ', 'cHW', 'cHd', 'cHe', 'cHu', 'cHud', 'cT', 'cWW', 'cd', 'cdB', 'cdG', 'cdW', 'cl', 'clB', 'clW', 'cpHL', 'cpHQ', 'cu', 'cuB', 'cuG', 'cuW', 'tc3G', 'tc3W', 'tcA', 'tcG', 'tcHB', 'tcHW']
coefficients = ['cuW', 'cuB', 'tc3G', 'c3G', 'cHu', 'c2G', 'cuG']
coefficients = ['cuW', 'cuB', 'cH', 'tc3G', 'c3G', 'cHu', 'c2G', 'cuG']
processes = [x.replace('process_cards/', '').replace('.dat', '') for x in glob.glob('process_cards/*.dat')]
constraints = ['ttZ', 'ttH', 'ttW']  # these processes are used to constrain the left and right scan bounds
scale = 30  # min and max scan bounds set according to NP / SM < scale for any of the constraint processes
sm_gridpack = '/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/madgraph/V5_2.3.2.2/ttZ01j_5f_MLM/v1/ttZ01j_5f_MLM_tarball.tar.xz'  # SM gridpack to clone
lhapdf = '/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/lhapdf/6.1.6/share/LHAPDF/../../bin/lhapdf-config'
madgraph = 'MG5_aMC_v2_3_3.tar.gz'  # madgraph tarball
np_model = 'HEL_UFO.third_gen.tar.gz'  # NP model tarball
np_param_path = 'HEL_UFO/restrict_no_b_mass.dat'  # path (relative to the unpackd NP model tarball) to the NP parameter card
sm_param_path = 'mgbasedir/models/sm/restrict_no_b_mass.dat'  # path (relative to the unpacked SM gridpack) to the SM parameter card
sm_card_path = 'process/madevent/Cards'  # path (relative to the unpacked SM gridpack) to the SM card directory
cores = 2
events = 100000
cutoff = (4 * np.pi) ** 2  # convergence of the loop expansion requires c < (4 * pi)^2, see section 7 https://arxiv.org/pdf/1205.4231.pdf
low = -1. * cutoff
high = 1. * cutoff
scale_numvalues = 25  # number of values to use for scale scan
interval_numvalues = 10  # number of values to use for interval scan
chunksize = 5
maxchunks = 1000

base = os.path.dirname(os.path.abspath(__file__))
cards = os.path.join(base, 'cards', version)
release = base[:base.find('/src')]

processes = constraints

# make a copy of the cards used corresponding to this run, otherwise it is confusing to keep track of changes
utils = imp.load_source('', os.path.join(base, '../python/utils.py'))
utils.clone_cards(
    os.path.join(base, sm_gridpack),
    os.path.join(base, np_model),
    sm_param_path,
    np_param_path,
    sm_card_path,
    cards,
    lhapdf
)

storage = StorageConfiguration(
    output=[
        "hdfs://eddie.crc.nd.edu:19000/store/user/$USER/" + version,
        "root://deepthought.crc.nd.edu//store/user/$USER/" + version,
        "srm://T3_US_NotreDame/store/user/$USER/" + version,
        "gsiftp://T3_US_NotreDame/store/user/$USER/" + version,
        "file:///hadoop/store/user/$USER/" + version,
        # "chirp://eddie.crc.nd.edu:9094/store/user/$USER/" + version,
    ],
    input=[
        # "root://deepthought.crc.nd.edu//store/user/$USER/{}/".format(version),
        "root://deepthought.crc.nd.edu//store/user/$USER/{}/".format('ttV/cross_sections/25'),
    ],
    disable_input_streaming=True
)

interval_resources = Category(
    name='interval',
    mode='min_waste',
    cores=cores,
    memory=2500,
    disk=1500
)

scale_resources = Category(
    name='scale',
    mode='min_waste',
    cores=cores,
    memory=2500,
    disk=1500
)


def chunk(size, numvalues, processes, coefficients, dim=1):
    requiredpoints = 1. + 2. * dim + (dim - 1.) * dim / 2.
    totalpoints = numvalues ** dim + 1
    if totalpoints < requiredpoints:
        raise ValueError('need more than {} points; {}d fit requires at least {}'.format(totalpoints, dim, requiredpoints))
    if size > totalpoints:
        size = totalpoints
    res = []
    sm = []
    for coefficient_group in utils.sorted_combos(coefficients, dim):
        for p in processes:
            unique_args = []
            for lower, higher in zip(np.arange(0, totalpoints, size), np.arange(size, totalpoints + 1, size)):
                unique_args += ['{} {}.dat {}'.format(','.join(coefficient_group), p, ' '.join([str(x) for x in np.arange(lower, higher)]))]
            sm += [unique_args.pop(0)]  # the first entry contains the SM point-- make sure we do not truncate it
            if higher < totalpoints:
                unique_args += ['{} {}.dat {}'.format(','.join(coefficient_group), p, ' '.join([str(x) for x in np.arange(higher, totalpoints)]))]
            np.random.shuffle(unique_args)
            unique_args = unique_args[:maxchunks]
            res += unique_args
    np.random.shuffle(res)

    return sm + res


interval = Workflow(
    label='interval',
    dataset=EmptyDataset(),
    category=interval_resources,
    sandbox=cmssw.Sandbox(release=release),
    command='python interval.py {nv} {cores} {events} {mg} {model} {pp} {cards} {low} {high}'.format(
        nv=interval_numvalues,
        cores=cores,
        events=events,
        mg=madgraph,
        model=np_model,
        pp=np_param_path,
        cards=os.path.split(cards)[-1],
        low=low,
        high=high),
    unique_arguments=chunk(chunksize, interval_numvalues, constraints, coefficients),
    merge_command='merge_scans',
    merge_size='2G',
    extra_inputs=[
        os.path.join(base, madgraph),
        os.path.join(base, np_model),
        cards,
        '{}/interval.py'.format(base)
    ] + ['{b}/process_cards/{p}.dat'.format(b=base, p=p) for p in constraints],
    outputs=['cross_sections.npz']
)

scale = Workflow(
        label='scale',
        dataset=Dataset('all/all.npz'),
        # dataset=Dataset('interval/merged.npz'),
        # dataset=ParentDataset(
        #     parent=interval,
        #     units_per_task=1
        # ),
        category=scale_resources,
        sandbox=cmssw.Sandbox(release=release),
        command='python scale.py {cv} {cores} {events} {mg} {model} {pp} {cards} {scale}'.format(
            cv=scale_numvalues,
            cores=cores,
            events=events,
            mg=madgraph,
            model=np_model,
            pp=np_param_path,
            cards=os.path.split(cards)[-1],
            scale=scale),
        unique_arguments=chunk(chunksize, scale_numvalues, processes, coefficients, dimension),
        merge_command='merge_scans',
        merge_size='2G',
        # merge_maxinputs=50,
        # cleanup_input=True,
        extra_inputs=[
            os.path.join(base, madgraph),
            os.path.join(base, np_model),
            cards,
            '{}/scale.py'.format(base),
            '/afs/crc.nd.edu/user/a/awoodard/.local/lib/python2.7/site-packages/tabulate.py'
        ] + ['{b}/process_cards/{p}.dat'.format(b=base, p=p) for p in processes],
        outputs=['cross_sections.npz']
    )

if 'email' in dir():
    options = AdvancedOptions(log_level=1, abort_multiplier=100000, email=email, bad_exit_codes=[42])
else:
    options = AdvancedOptions(log_level=1, abort_multiplier=100000, bad_exit_codes=[42], dashboard=False)

units = sum([len(w.unique_arguments) for w in [interval, scale]])
print 'will make {} units'.format(units)

config = Config(
    label=str(version).replace('/', '_'),
    workdir='/tmpscratch/users/$USER/' + version,
    plotdir='~/www/lobster/' + version,
    storage=storage,
    # workflows=[interval, scale],
    workflows=[scale],
    advanced=options
)
