'''
to start a factory:
nohup work_queue_factory -T condor -M lobster_$USER.*ttV.*gridpacks -d all -o /tmp/${USER}_lobster_ttV_gridpack.debug -C $(readlink -f gridpack_factory.json) >& /tmp/${USER}_lobster_ttV_gridpack.log &
'''
import datetime
import os
import json

import numpy as np

from lobster import cmssw
from lobster.core import *

operators = ['c2B', 'c2G', 'c2W', 'c3G', 'c3W', 'c6', 'cA', 'cB', 'cG', 'cH', 'cHB', 'cHL', 'cHQ', 'cHW', 'cHd', 'cHe', 'cHu', 'cHud', 'cT', 'cWW', 'cd', 'cdB', 'cdG', 'cdW', 'cl', 'clB', 'clW', 'cpHL', 'cpHQ', 'cu', 'cuB', 'cuG', 'cuW', 'tc3G', 'tc3W', 'tcA', 'tcG', 'tcHB', 'tcHW']
# operators = ['c2W', 'c3G', 'c3W', 'cA', 'cB', 'cG', 'cHB', 'cHQ', 'cHW',
#              'cHd', 'cHu', 'cHud', 'cT', 'cWW', 'cpHQ', 'cu', 'cuB',
#              'cuG', 'cuW', 'tc3G', 'tc3W', 'tcG', 'tcHW']
# operators = ['cuB', 'cpHQ', 'cHQ', 'cHu', 'c3W']

version = 'ttV/36'
base = os.path.dirname(os.path.abspath(__file__))
release = base[:base.find('/src')]
points_file = '/afs/crc.nd.edu/user/a/awoodard/releases/effective-ttV-production/CMSSW_7_4_7/src/EffectiveTTVProduction/EffectiveTTVProduction/test/linspace_30_points.npy'
points = np.load(points_file)[()]

storage = StorageConfiguration(
    output=[
        "hdfs://eddie.crc.nd.edu:19000/store/user/$USER/" + version,
        # "file:///hadoop/store/user/$USER/" + version,
        "root://deepthought.crc.nd.edu//store/user/$USER/" + version,
        "srm://T3_US_NotreDame/store/user/$USER/" + version,
        "gsiftp://T3_US_NotreDame/store/user/$USER/" + version,
        # "chirp://eddie.crc.nd.edu:9094/store/user/$USER/" + version,
    ]
)

processing = Category(
    name='processing',
    cores=2,
    memory=1000,
    disk=2000
)

workflows = []

# for operator in operators:
for operator in ['cHu', 'cu', 'cuB', 'cuW']:
    # for process in ['WW']:
    # FIXME add tHq and tWH
    # for process in ['DY', 'H', 'WWW', 'WWZ', 'WZ', 'WZZ', 'ZZ', 'ZZZ', 'tZq', 'tt', 'ttH', 'ttW', 'ttZ', 'WW', 'tttt', 'tWZ']:
    for process in ['ttH', 'ttW', 'ttZ']:
        workflows.append(Workflow(
            label='{}_gridpacks_{}'.format(process, operator),
            dataset=EmptyDataset(number_of_tasks=1),
            category=processing,
            sandbox=cmssw.Sandbox(release=release),
            command='python {base}/clone_tarball.py {base}/process_cards/{process}.dat /cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/madgraph/V5_2.3.2.2/ttZ01j_5f_MLM/v1/ttZ01j_5f_MLM_tarball.tar.xz {base}/MG5_aMC_v2_3_3.third_gen.tar.gz mgbasedir/models/sm/restrict_no_b_mass.dat models/HEL_UFO/restrict_no_b_mass.dat 1 {ops} {points}'.format(base=base, process=process, ops=operator, points=points_file),
            unique_arguments=range(0, len(points.values()[0])),
            outputs=['gridpack.tar.xz']
            )
        )
        workflows.append(Workflow(
            label='{}_diagrams_{}'.format(process, operator),
            dataset=EmptyDataset(number_of_tasks=1),
            category=processing,
            sandbox=cmssw.Sandbox(release=release),
            command='python {base}/clone_tarball.py {base}/process_cards/{process}.dat /cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/madgraph/V5_2.3.2.2/ttZ01j_5f_MLM/v1/ttZ01j_5f_MLM_tarball.tar.xz {base}/MG5_aMC_v2_3_3.third_gen.tar.gz mgbasedir/models/sm/restrict_no_b_mass.dat models/HEL_UFO/restrict_no_b_mass.dat 1 {ops} {points}'.format(base=base, process=process, ops=operator, points=points_file),
            unique_arguments=[0],
            outputs=['diagrams.tar.xz']
            )
        )
config = Config(
    label=str(version).replace('/', '_') + '_gridpacks',
    workdir='/tmpscratch/users/$USER/' + version,
    plotdir='~/www/lobster/' + version,
    storage=storage,
    workflows=workflows,
    advanced=AdvancedOptions(log_level=1, abort_multiplier=100000, dashboard=False)
)
