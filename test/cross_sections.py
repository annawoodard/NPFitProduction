import glob
import imp
import itertools
import os

import numpy as np

from lobster import cmssw
from lobster.core import *

# corresponds to commit 9055a1425ef6bd7c238fd9300f21a9589e6e1057

# email = 'changeme@changeme.edu'  # uncomment to have notification sent here when processing completes
dimension = 2  # number of coefficients to change per scan
version = 'ttV/cross_sections/12/{}d'.format(dimension)  # you should increment this each time you make changes
# coefficients = ['c2B', 'c2G', 'c2W', 'c3G', 'c3W', 'c6', 'cA', 'cB', 'cG', 'cH', 'cHB', 'cHL', 'cHQ', 'cHW', 'cHd', 'cHe', 'cHu', 'cHud', 'cT', 'cWW', 'cd', 'cdB', 'cdG', 'cdW', 'cl', 'clB', 'clW', 'cpHL', 'cpHQ', 'cu', 'cuB', 'cuG', 'cuW', 'tc3G', 'tc3W', 'tcA', 'tcG', 'tcHB', 'tcHW']
coefficients = ['cuW', 'cuB', 'cH', 'tc3G', 'c3G', 'cHu', 'c2G', 'cuG']
processes = [x.replace('process_cards/', '').replace('.dat', '') for x in glob.glob('process_cards/*.dat')]
constraints = ['ttZ', 'ttH', 'ttW']  # these processes are used to constrain the left and right scan bounds
scale = 10  # min and max scan bounds set according to NP / SM < scale for any of the constraint processes
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
low = -1. * cutoff
high = 1. * cutoff
zooms = 0  # number of iterations to make for zooming in on the wilson coefficient range corresponding to NP / SM < scale
final_numvalues = 30  # number of values to use for final scan
zoom_numvalues = 3  # number of values to use for each zoom scan
interpolate_numvalues = 1000  # number of values to use for interpolating
chunksize = 10
maxchunks = 1000

base = os.path.dirname(os.path.abspath(__file__))
cards = os.path.join(base, 'cards', version)
release = base[:base.find('/src')]

processes = constraints

#  make a copy of the cards used corresponding to this run, otherwise it is confusing to keep track of changes
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
        "root://deepthought.crc.nd.edu//store/user/$USER/{}/".format(version),
    ],
    disable_input_streaming=True
)

madgraph_resources = Category(
    name='madgraph',
    cores=cores,
    memory=4000,
    disk=2500
)


def chunk(size, numvalues, processes, coefficients):
    dim = len(coefficients)
    requiredpoints = 1. + 2. * dim + (dim - 1.) * dim / 2.
    totalpoints = numvalues ** dim
    if totalpoints < requiredpoints:
        raise ValueError('need more than {} points; {}d fit requires at least {}'.format(totalpoints, dim, requiredpoints))
    if size > totalpoints:
        size = totalpoints
    unique_args = []
    for p in processes:
        for lower, higher in zip(np.arange(0, totalpoints, size), np.arange(size, totalpoints + 1, size)):
            unique_args += ['{}.dat {}'.format(p, ' '.join([str(x) for x in np.arange(lower, higher)]))]
    unique_args = unique_args[:maxchunks]
    np.random.shuffle(unique_args)
    return unique_args


workflows = []
for coefficient_group in itertools.combinations(coefficients, dimension):
    tag = '_'.join(coefficient_group)
    zoom = Workflow(
        label='interval_pass_{}'.format(tag),
        dataset=EmptyDataset(),
        category=madgraph_resources,
        sandbox=cmssw.Sandbox(release=release),
        command='python interval.py {nv} {cores} {coefficients} {events} {mg} {model} {pp} {cards} {low} {high}'.format(
            nv=zoom_numvalues,
            cores=cores,
            coefficients=','.join(coefficient_group),
            events=events,
            mg=madgraph,
            model=np_model,
            pp=np_param_path,
            cards=os.path.split(cards)[-1],
            low=low,
            high=high),
        unique_arguments=chunk(chunksize, zoom_numvalues, constraints, coefficient_group),
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
    workflows.append(zoom)

    for i in range(zooms):
        # A fit to the interval scan above is used to find the range of Wilson coefficient
        # values corresponding to `scale`. The fit may not perform well at very
        # large Wilson coefficient values. Here we use a few iterations,
        # each time reducing the scale. This allows the fit to 'settle down' as the
        # target scale is approached. Multiple zooms are not usually necessary for range
        # finding in reasonable intervals.
        zoom = Workflow(
            label='zoom_pass_{}_{}'.format(i + 1, tag),
            dataset=ParentDataset(
                parent=zoom,
                units_per_task=1
            ),
            category=madgraph_resources,
            sandbox=cmssw.Sandbox(release=release),
            command='python scale.py {iv} {cv} {cores} {coefficients} {events} {mg} {model} {pp} {cards} {scale}'.format(
                iv=interpolate_numvalues,
                cv=zoom_numvalues,
                cores=cores,
                coefficients=','.join(coefficient_group),
                events=events,
                mg=madgraph,
                model=np_model,
                pp=np_param_path,
                cards=os.path.split(cards)[-1],
                scale=scale * (zooms + 1 - i)),
            unique_arguments=chunk(chunksize, zoom_numvalues, constraints, coefficient_group),
            merge_command='merge_scans',
            merge_size='2G',
            # cleanup_input=True,
            extra_inputs=[
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
            command='python scale.py {ip} {cv} {cores} {coefficients} {events} {mg} {model} {pp} {cards} {scale}'.format(
                ip=interpolate_numvalues,
                cv=final_numvalues,
                cores=cores,
                coefficients=','.join(coefficient_group),
                events=events,
                mg=madgraph,
                model=np_model,
                pp=np_param_path,
                cards=os.path.split(cards)[-1],
                scale=scale),
            unique_arguments=chunk(chunksize, final_numvalues, processes, coefficient_group),
            merge_command='merge_scans',
            merge_size='2G',
            # cleanup_input=True,
            extra_inputs=[
                os.path.join(base, madgraph),
                os.path.join(base, np_model),
                cards,
                '{}/scale.py'.format(base)
            ] + ['{b}/process_cards/{p}.dat'.format(b=base, p=p) for p in processes],
            outputs=['cross_sections.npz']
        )
    )

if 'email' in dir():
    options = AdvancedOptions(log_level=1, abort_multiplier=100000, email=email, bad_exit_codes=[42])
else:
    options = AdvancedOptions(log_level=1, abort_multiplier=100000, bad_exit_codes=[42])

units = sum([len(w.unique_arguments) for w in workflows])
print 'will make {} units'.format(units)

config = Config(
    label=str(version).replace('/', '_'),
    workdir='/tmpscratch/users/$USER/' + version,
    plotdir='~/www/lobster/' + version,
    storage=storage,
    workflows=workflows,
    advanced=options
)
