'''
to start a factory:
nohup work_queue_factory -T condor -M lobster_$USER.*ttV.*gen -d all -o /tmp/${USER}_lobster_ttV_gen.debug -C $(readlink -f gen_factory.json) >& /tmp/${USER}_lobster_ttV_gen.log &
'''
import datetime
import glob
import os

from lobster import cmssw
from lobster.core import *
from lobster.monitor.elk.interface import ElkInterface

gridpack_version = 'spam-1'
gen_version = '1'
version = 'ttV/{}/{}'.format(gridpack_version, gen_version)
email = 'awoodard@nd.edu'
base = os.path.dirname(os.path.abspath(__file__))
cards = os.path.join(base, 'cards', version)
release = base[:base.find('/src')]
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
numpoints = 30
chunksize = 10  # number of points to calculate per task

coefficients = ['cHu', 'cu', 'cuW', 'cuB']
processes = ['ttZ']
zoom_numpoints = 10
# dimension = 2
chunksize = 10

utils = imp.load_source('', os.path.join(base, '../python/utils.py'))
utils.clone_cards(
    sm_gridpack,
    np_model,
    sm_param_path,
    np_param_path,
    sm_card_path,
    cards,
    lhapdf,
    extras=['mgbasedir', 'runcmsgrid.sh']
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

workflows = []
for process in processes[-1]:
    for coefficient_group in itertools.combinations(coefficients, dimension):
        unique_args = []
        for low, high in zip(np.arange(0, numpoints, chunksize), np.arange(chunksize, numpoints + chunksize, chunksize)):
            unique_args += [range(low, high)]
        tag = '_'.join(coefficient_group)
        gridpacks = Workflow(
            label='{}_gridpacks_{}'.format(process, tag),
            dataset=MultiGridpackDataset(events_per_gridpack=26000, events_per_task=13000),
            category=gridpack_resources,
            sandbox=cmssw.Sandbox(release=release),
            command='python cross_sections.py scale {np} {cores} {coefficients} {events} {mg} {model} {pp} {cards} {scale}'.format(
                np=cross_section_numpoints,
                cores=cores,
                coefficients=','.join(coefficient_group),
                events=events,
                mg='MG5_aMC_v2_3_3.tar.gz',
                model='HEL_UFO.third_gen.tar.gz',
                pp=np_param_path,
                cards=os.path.split(cards)[-1],
                scale=scale),
            unique_arguments=range(0, len(points.values()[0])),
            extra_inputs=[
                tempdir.__file__,
                '{}/MG5_aMC_v2_3_3.tar.gz'.format(base),
                '{}/HEL_UFO.third_gen.tar.gz'.format(base),
                cards,
                '{b}/process_cards/{p}.dat'.format(b=base, p=p)
            ]
            outputs=['gridpack.tar.xz']
        )

        lhe = Workflow(
                label='{}_lhe_{}'.format(process, tag),
                pset='HIG-RunIIWinter15wmLHE-01035_1_cfg.py',
                sandbox=cmssw.Sandbox(release=release),
                outputs=['HIG-RunIIWinter15wmLHE-01035ND.root'],
                globaltag=False,
                dataset=ParentDataset(parent=gridpacks),
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

config = Config(
    label=str(version).replace('/', '_') + '_gen',
    workdir='/tmpscratch/users/$USER/' + version,
    plotdir='~/www/lobster/' + version,
    storage=storage,
    workflows=workflows,
    advanced=AdvancedOptions(log_level=1, bad_exit_codes=[127], abort_multiplier=100000, email=email)
    # elk=ElkInterface('elk.crc.nd.edu', 9200, 'elk.crc.nd.edu', 5601, project='ttV.{}.{}'.format(gridpack_version, gen_version), dashboards=['Core', 'Advanced', 'Tasks'])
)
