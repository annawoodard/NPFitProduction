'''
to start a factory:
nohup work_queue_factory -T condor -M lobster_$USER.*ttV.*gridpacks -d all -o /tmp/${USER}_lobster_ttV_gridpack.debug -C $(readlink -f gridpack_factory.json) >& /tmp/${USER}_lobster_ttV_gridpack.log &
'''
import datetime
import os

from lobster import cmssw
from lobster.core import *

# from EffectiveTTVProduction.EffectiveTTVProduction.operators import operators
operators = ['c2W', 'c3G', 'c3W', 'cA', 'cB', 'cG', 'cHB', 'cHQ', 'cHW',
             'cHd', 'cHu', 'cHud', 'cT', 'cWW', 'cpHQ', 'cu', 'cuB',
             'cuG', 'cuW', 'tc3G', 'tc3W', 'tcG', 'tcHW']

version = 'ttV/5'
base = os.path.dirname(os.path.abspath(__file__))
release = base[:base.find('/src')]
points = 200

storage = StorageConfiguration(
    output=[
        "hdfs://eddie.crc.nd.edu:19000/store/user/$USER/" + version,
        "file:///hadoop/store/user/$USER/" + version,
        "root://deepthought.crc.nd.edu//store/user/$USER/" + version,
        "chirp://eddie.crc.nd.edu:9094/store/user/$USER/" + version,
        "gsiftp://T3_US_NotreDame/store/user/$USER/" + version,
        "srm://T3_US_NotreDame/store/user/$USER/" + version
    ]
)

processing = Category(
    name='processing',
    cores=2,
    memory=1000
)

workflows = []

for operator in operators:
    workflows.append(Workflow(
        label='ttW_gridpacks_{}'.format(operator),
        dataset=EmptyDataset(number_of_tasks=1),
        category=processing,
        sandbox=cmssw.Sandbox(release=release),
        command='python {base}/clone_tarball.py proc_card_ttW.dat /cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/madgraph/V5_2.3.2.2/ttW012j_5f_madgraph_mlmMerging/v1/ttW012j_5f_madgraph_mlmMerging_tarball.tar.xz {base}/MG5_aMC_v2_3_3.tar.gz mgbasedir/models/sm/restrict_no_b_mass.dat models/HEL_UFO/restrict_no_b_mass.dat 1 {ops}'.format(base=base, ops=operator),
        unique_arguments=range(0, points),
        outputs=['gridpack.tar.xz'],
        extra_inputs=['linspace_points.json', 'proc_card_ttW.dat']
        )
    )

    workflows.append(Workflow(
        label='ttZ_gridpacks_{}'.format(operator),
        dataset=EmptyDataset(number_of_tasks=1),
        category=processing,
        sandbox=cmssw.Sandbox(release=release),
        command='python {base}/clone_tarball.py proc_card_ttZ.dat /cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/madgraph/V5_2.3.2.2/ttZ01j_5f_MLM/v1/ttZ01j_5f_MLM_tarball.tar.xz {base}/MG5_aMC_v2_3_3.tar.gz mgbasedir/models/sm/restrict_no_b_mass.dat models/HEL_UFO/restrict_no_b_mass.dat 1 {ops}'.format(base=base, ops=operator),
        unique_arguments=range(0, points),
        outputs=['gridpack.tar.xz'],
        extra_inputs=['linspace_points.json', 'proc_card_ttZ.dat']
        )
    )

    workflows.append(Workflow(
        label='ttH_gridpacks_{}'.format(operator),
        dataset=EmptyDataset(number_of_tasks=1),
        category=processing,
        sandbox=cmssw.Sandbox(release=release),
        command='python {base}/clone_tarball.py proc_card_ttH.dat /cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/madgraph/V5_2.3.2.2/ttZ01j_5f_MLM/v1/ttZ01j_5f_MLM_tarball.tar.xz {base}/MG5_aMC_v2_3_3.tar.gz mgbasedir/models/sm/restrict_no_b_mass.dat models/HEL_UFO/restrict_no_b_mass.dat 1 {ops}'.format(base=base, ops=operator),
        unique_arguments=range(0, points),
        outputs=['gridpack.tar.xz'],
        extra_inputs=['linspace_points.json', 'proc_card_ttH.dat']
        )
    )

config = Config(
    label=str(version).replace('/', '_') + '_gridpacks',
    workdir='/tmpscratch/users/$USER/' + version,
    plotdir='~/www/lobster/' + version,
    storage=storage,
    workflows=workflows,
    advanced=AdvancedOptions(log_level=1, use_dashboard=False),
)
