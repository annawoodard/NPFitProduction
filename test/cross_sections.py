'''
to start a factory on the ND T3:
nohup work_queue_factory -T condor -M lobster_$USER.*ttV.*cross_sections.* -d all -o /tmp/${USER}_lobster_ttV_factory.debug -C $(readlink -f factory_cross_sections_T3.json) >& /tmp/${USER}_lobster_ttV_xsec.log &

to start a factory at the CRC:
nohup work_queue_factory -T condor -M lobster_$USER.*ttV.*cross_sections.* -d all -o /tmp/${USER}_lobster_ttV_factory.debug -C $(readlink -f factory_cross_sections_CRC.json) --wrapper "python /afs/crc.nd.edu/group/ccl/software/runos/runos.py rhel6" --extra-options="--workdir=/disk" --worker-binary=/afs/crc.nd.edu/group/ccl/software/x86_64/redhat6/cctools/$cctools/bin/work_queue_worker >&  /tmp/${USER}_lobster_ttV_xsec.log &
'''
import datetime
import glob
import imp
import itertools
import json
import os
import tempdir

import numpy as np

from lobster import cmssw
from lobster.core import *

# email = 'changeme@changeme.edu'  # uncomment to have notification sent here when processing completes
# email = 'awoodard@nd.edu'  # notification will be sent here when processing completes
version = 'ttV/cross_sections/spam-34'
coefficients = ['c2B', 'c2G', 'c2W', 'c3G', 'c3W', 'c6', 'cA', 'cB', 'cG', 'cH', 'cHB', 'cHL', 'cHQ', 'cHW', 'cHd', 'cHe', 'cHu', 'cHud', 'cT', 'cWW', 'cd', 'cdB', 'cdG', 'cdW', 'cl', 'clB', 'clW', 'cpHL', 'cpHQ', 'cu', 'cuB', 'cuG', 'cuW', 'tc3G', 'tc3W', 'tcA', 'tcG', 'tcHB', 'tcHW']
processes = [x.replace('process_cards/', '').replace('.dat', '') for x in glob.glob('process_cards/*.dat')]
constraints = ['ttZ', 'ttH', 'ttW']  # these processes are used to constrain the left and right scan bounds
scale = 5  # min and max scan bounds set according to NP / SM < scale for any of the constraint processes
sm_gridpack = '/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/madgraph/V5_2.3.2.2/ttZ01j_5f_MLM/v1/ttZ01j_5f_MLM_tarball.tar.xz'  # SM gridpack to clone
lhapdf = '/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/lhapdf/6.1.6/share/LHAPDF/../../bin/lhapdf-config'
madgraph = 'MG5_aMC_v2_3_3.tar.gz'  # madgraph tarball
np_model = 'HEL_UFO.third_gen.tar.gz'  # NP model tarball
np_param_path = 'HEL_UFO/restrict_no_b_mass.dat'  # path (relative to the unpackd NP model tarball) to the NP parameter card
sm_param_path = 'mgbasedir/models/sm/restrict_no_b_mass.dat'  # path (relative to the unpacked SM gridpack) to the SM parameter card
sm_card_path = 'process/madevent/Cards'  # path (relative to the unpacked SM gridpack) to the SM card directory
cores = 12
events = 50000
cutoff = (4 * np.pi) ** 2  # convergence of the loop expansion requires c < (4 * pi)^2, see section 7 https://arxiv.org/pdf/1205.4231.pdf
dimension = 1  # number of coefficients to change per scan
cross_section_numpoints = 30  # number of points to run for final scan
chunksize = 10  # number of points to calculate per task
zooms = 2  # number of iterations to make for zooming in on the wilson coefficient range corresponding to NP / SM < scale
zoom_numpoints = 100  # number of points to run for each zoom scan

base = os.path.dirname(os.path.abspath(__file__))
cards = os.path.join(base, 'cards', version)
release = base[:base.find('/src')]

#  make a copy of the cards used corresponding to this run, otherwise it is confusing to keep track of changes
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

madgraph_resources = Category(
    name='madgraph',
    cores=cores,
    memory=4000,
    disk=4000
)


def chunk(size, numpoints, processes, coefficients):
    unique_args = []
    if size > numpoints:
        size = numpoints
    for p in processes:
        totalpoints = numpoints * len(coefficients)
        for low, high in zip(np.arange(0, totalpoints, size), np.arange(size, totalpoints + size, size)):
            unique_args += ['{}.dat {}'.format(p, ' '.join([str(x) for x in np.arange(low, high)]))]
    return unique_args

workflows = []
for coefficient_group in itertools.combinations(coefficients, dimension):
    tag = '_'.join(coefficient_group)
    zoom = Workflow(
        label='zoom_pass_1_{}'.format(tag),
        dataset=EmptyDataset(),
        category=madgraph_resources,
        sandbox=cmssw.Sandbox(release=release),
        command='python interval.py {np} {cores} {coefficients} {events} {mg} {model} {pp} {cards} {low} {high}'.format(
            np=zoom_numpoints,
            cores=cores,
            coefficients=','.join(coefficient_group),
            events=events,
            mg=madgraph,
            model=np_model,
            pp=np_param_path,
            cards=os.path.split(cards)[-1],
            low=-1. * cutoff,
            high=cutoff),
        unique_arguments=chunk(chunksize, zoom_numpoints, constraints, coefficient_group),
        merge_command='merge_scans',
        merge_size='2G',
        extra_inputs=[
            tempdir.__file__,
            os.path.join(base, madgraph),
            os.path.join(base, np_model),
            cards,
            '{}/interval.py'.format(base)
        ] + ['{b}/process_cards/{p}.dat'.format(b=base, p=p) for p in constraints],
        outputs=['cross_sections.npz']
    )
    workflows.append(zoom)

    for i in range(zooms - 1):
        zoom = Workflow(
            label='zoom_pass_{}_{}'.format(i + 2, tag),
            dataset=ParentDataset(
                parent=zoom,
                units_per_task=1
            ),
            category=madgraph_resources,
            sandbox=cmssw.Sandbox(release=release),
            command='python scale.py {np} {cores} {coefficients} {events} {mg} {model} {pp} {cards} {scale}'.format(
                np=zoom_numpoints,
                cores=cores,
                coefficients=','.join(coefficient_group),
                events=events,
                mg=madgraph,
                model=np_model,
                pp=np_param_path,
                cards=os.path.split(cards)[-1],
                scale=scale),
            unique_arguments=chunk(chunksize / 2, zoom_numpoints, constraints, coefficient_group),
            merge_command='merge_scans',
            merge_size='2G',
            cleanup_input=True,
            extra_inputs=[
                tempdir.__file__,
                os.path.join(base, madgraph),
                os.path.join(base, np_model),
                cards,
                '{}/scale.py'.format(base)
            ] + ['{b}/process_cards/{p}.dat'.format(b=base, p=p) for p in constraints],
            outputs=['cross_sections.npz']
        )
        workflows.append(zoom)

    workflows.append(
        Workflow(
            label='final_pass_{}'.format(tag),
            dataset=ParentDataset(
                parent=zoom,
                units_per_task=1
            ),
            category=madgraph_resources,
            sandbox=cmssw.Sandbox(release=release),
            command='python scale.py {np} {cores} {coefficients} {events} {mg} {model} {pp} {cards} {scale}'.format(
                np=cross_section_numpoints,
                cores=cores,
                coefficients=','.join(coefficient_group),
                events=events,
                mg=madgraph,
                model=np_model,
                pp=np_param_path,
                cards=os.path.split(cards)[-1],
                scale=scale),
            unique_arguments=chunk(chunksize / 2, cross_section_numpoints, processes, coefficient_group),
            merge_command='merge_scans',
            merge_size='2G',
            cleanup_input=True,
            extra_inputs=[
                tempdir.__file__,
                os.path.join(base, madgraph),
                os.path.join(base, np_model),
                cards,
                '{}/scale.py'.format(base)
            ] + ['{b}/process_cards/{p}.dat'.format(b=base, p=p) for p in processes],
            outputs=['cross_sections.npz']
        )
    )

if 'email' in dir():
    options = AdvancedOptions(log_level=1, abort_multiplier=100000, email=email)
else:
    options = AdvancedOptions(log_level=1, abort_multiplier=100000)

config = Config(
    label=str(version).replace('/', '_'),
    workdir='/tmpscratch/users/$USER/' + version,
    plotdir='~/www/lobster/' + version,
    storage=storage,
    workflows=workflows,
    advanced=options
)
