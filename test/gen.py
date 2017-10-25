import datetime
import glob
import imp
import itertools
import os
import tempdir

import numpy as np

from lobster import cmssw
from lobster.core import *

version = 'ttV/gen/1'
# email = 'changeme@changeme.edu'  # notification will be sent here when processing completes
coefficients = ['c2B', 'c2G', 'c2W', 'c3G', 'c3W', 'c6', 'cA', 'cB', 'cG', 'cH', 'cHB', 'cHL', 'cHQ', 'cHW', 'cHd', 'cHe', 'cHu', 'cHud', 'cT', 'cWW', 'cd', 'cdB', 'cdG', 'cdW', 'cl', 'clB', 'clW', 'cpHL', 'cpHQ', 'cu', 'cuB', 'cuG', 'cuW', 'tc3G', 'tc3W', 'tcA', 'tcG', 'tcHB', 'tcHW']
sm_gridpack = '/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/madgraph/V5_2.3.2.2/ttZ01j_5f_MLM/v1/ttZ01j_5f_MLM_tarball.tar.xz'  # SM gridpack to clone
processes = [x.replace('process_cards/', '').replace('.dat', '') for x in glob.glob('process_cards/*.dat')]
lhapdf = '/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/lhapdf/6.1.6/share/LHAPDF/../../bin/lhapdf-config'
madgraph = 'MG5_aMC_v2_3_3.tar.gz'  # madgraph tarball
np_model = 'HEL_UFO.third_gen.tar.gz'  # NP model tarball
np_param_path = 'HEL_UFO/restrict_no_b_mass.dat'  # path (relative to the unpackd NP model tarball) to the NP parameter card
sm_param_path = 'mgbasedir/models/sm/restrict_no_b_mass.dat'  # path (relative to the unpacked SM gridpack) to the SM parameter card
sm_card_path = 'process/madevent/Cards'  # path (relative to the unpacked SM gridpack) to the SM card directory
cores = 12
events = 50000
dimension = 1  # number of coefficients to change per scan
numvalues = 30  # number of values per coefficient

base = os.path.dirname(os.path.abspath(__file__))
cards = os.path.join(base, 'cards', version)
release = base[:base.find('/src')]

####### needed for interval rangefinding method (otherwise, these have no effect) #######
cutoff = (4 * np.pi) ** 2  # convergence of the loop expansion requires c < (4 * pi)^2, see section 7 https://arxiv.org/pdf/1205.4231.pdf
# low = -1. * cutoff
# high = 1. * cutoff
low = -10
high = 10
#########################################################################################

####### needed for scaling rangefinding method (otherwise, these have no effect) ########
scale = 5  # min and max scan bounds set according to NP / SM < scale for any of the constraint processes
scan = 'final_pass.total.npz'  # the merged output from running `test/cross_sections.py`, i.e. `merge_scans final_pass.total.npz outdir/final_pass_*/*npz`
cross_sections_version = 'ttV/cross_sections/1'  # the version from `test/cross_sections.py`
constraints = ['ttW', 'ttZ', 'ttH']  # processes to consider for range finding (none of these processes can exceed  NP / SM < scale)
#########################################################################################

# This function 'clones' an SM gridpack by copying its cards and all common parameters between it and
# the NP parameter card. If you want to use your own cards, comment this out and point the `cards` variable
# to a directory containing your run_card.dat, grid_card.dat, me5_configuration.txt, and the NP param_card.dat
utils = imp.load_source('', os.path.join(base, '../python/utils.py'))
utils.clone_cards(
    sm_gridpack,
    np_model,
    sm_param_path,
    np_param_path,
    sm_card_path,
    cards,
    lhapdf
)

storage = StorageConfiguration(
    output=[
        "hdfs://eddie.crc.nd.edu:19000/store/user/$USER/" + version,
        # "file:///hadoop/store/user/$USER/" + version,
        "root://deepthought.crc.nd.edu//store/user/$USER/" + version,
        "srm://T3_US_NotreDame/store/user/$USER/" + version,
        "gsiftp://T3_US_NotreDame/store/user/$USER/" + version,
        # "chirp://eddie.crc.nd.edu:9094/store/user/$USER/" + version,
    ],
    input=[
        "root://deepthought.crc.nd.edu//store/user/$USER/{}/".format(version),
    ],
    disable_input_streaming=True
)

gridpack_resources = Category(
    name='gridpack',
    cores=2,
    memory=1000,
    disk=2000
)

lhe_resources = Category(
    name='lhe',
    cores=2,
    memory=1500,
    disk=2000
)

gen_resources = Category(
    name='gen',
    cores=2,
    memory=2000,
    disk=1000,
)

gridpack_inputs=[
    tempdir.__file__,
    os.path.join(base, madgraph),
    os.path.join(base, np_model),
    '{}/gridpack.py'.format(base),
    cards
]

workflows = []
for process in processes:
    for coefficient_group in itertools.combinations(coefficients, dimension):
        tag = '_'.join(coefficient_group)
        gridpacks = Workflow(
            label='{}_gridpacks_{}'.format(process, tag),
            dataset=MultiGridpackDataset(events_per_gridpack=26000, events_per_lumi=13000),
            category=gridpack_resources,
            sandbox=cmssw.Sandbox(release=release),
            # Use the command and extra_inputs below to constrain coefficient values with an input scan and
            # scale value rather than an interval. You can obtain the input scan by running 'cross_sections.py'
            # and then (example assuming scan='final_pass.total.npz' and cross_sections_version='ttV/cross_sections/1')
            #     merge_scans final_pass.total.npz /hadoop/store/user/$USER/ttV/cross_sections/1/final_pass_*/*npz
            #     mv final_pass.total.npz /hadoop/store/user/$USER/ttV/cross_sections/1/
            # command='python gridpack.py {np} {cores} {coefficients} {events} {mg} {model} {pp} {cards} {pcard} --constraints {constraints} --scale {scale} --scan {scan}'.format(
            #     np=numvalues,
            #     cores=cores,
            #     coefficients=','.join(coefficient_group),
            #     events=events,
            #     mg=madgraph,
            #     model='HEL_UFO.third_gen.tar.gz',
            #     pp=np_param_path,
            #     cards=os.path.split(cards)[-1],
            #     pcard='{}.dat'.format(process),
            #     constraints=constraints
            #     scale=scale,
            #     scan=scan),
            # extra_inputs=gridpack_inputs + [
            #     '{b}/process_cards/{p}.dat'.format(b=base, p=process),
            #     os.path.join('/hadoop/store/user/$USER', cross_section_version, scan')
            # ]
            command='python gridpack.py {np} {cores} {coefficients} {events} {sm} {mg} {model} {pp} {cards} {pcard} --low {low} --high {high}'.format(
                np=numvalues,
                cores=cores,
                coefficients=','.join(coefficient_group),
                events=events,
                sm=sm_gridpack,
                mg=madgraph,
                model=np_model,
                pp=np_param_path,
                cards=os.path.split(cards)[-1],
                pcard='{}.dat'.format(process),
                low=low,
                high=high),
            extra_inputs=gridpack_inputs + ['{b}/process_cards/{p}.dat'.format(b=base, p=process)],
            unique_arguments=range(numvalues * len(coefficient_group)),
            outputs=['gridpack.tar.xz']
        )

        lhe = Workflow(
                label='{}_lhe_{}'.format(process, tag),
                pset='HIG-RunIIWinter15wmLHE-01035_1_cfg.py',
                sandbox=cmssw.Sandbox(release=release),
                outputs=['HIG-RunIIWinter15wmLHE-01035ND.root'],
                globaltag=False,
                dataset=ParentMultiGridpackDataset(parent=gridpacks, randomize_seeds=True),
                category=lhe_resources,
        )

        gen = Workflow(
                label='{}_gen_{}'.format(process, tag),
                pset='HIG-RunIISummer15Gen-01168_1_cfg.py',
                sandbox=cmssw.Sandbox(release=release),
                merge_size='3.5G',
                cleanup_input=True,
                outputs=['HIG-RunIISummer15Gen-01168ND.root'],
                globaltag=False,
                dataset=ParentDataset(parent=lhe),
                category=gen_resources
        )

        workflows.extend([gridpacks, lhe, gen])

if 'email' in dir():
    options = AdvancedOptions(log_level=1, bad_exit_codes=[127], abort_multiplier=100000, email=email)
else:
    options = AdvancedOptions(log_level=1, bad_exit_codes=[127], abort_multiplier=100000)

config = Config(
    label=str(version).replace('/', '_') + '_gen',
    workdir='/tmpscratch/users/$USER/' + version,
    plotdir='~/www/lobster/' + version,
    storage=storage,
    workflows=workflows,
    advanced=options
)
