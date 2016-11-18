'''
to start a factory:
nohup work_queue_factory -T condor -M lobster_$USER.*ttV.*gen -d all -o /tmp/${USER}_lobster_ttV_gen.debug -C $(readlink -f gen_factory.json) >& /tmp/${USER}_lobster_ttV_gen.log &
'''
import datetime
import os

from lobster import cmssw
from lobster.core import *
from lobster.monitor.elk.interface import ElkInterface

gridpack_version = '7'
gen_version = '1'
base = os.path.dirname(os.path.abspath(__file__))
release = base[:base.find('/src')]

# from EffectiveTTVProduction.EffectiveTTVProduction.operators import operators
operators = ['c2W', 'c3G', 'c3W', 'cA', 'cB', 'cG', 'cHB', 'cHQ', 'cHW',
             'cHd', 'cHu', 'cHud', 'cT', 'cWW', 'cpHQ', 'cu', 'cuB',
             'cuG', 'cuW', 'tc3G', 'tc3W', 'tcG', 'tcHW']
# operators = ['cuB', 'cpHQ', 'cHQ', 'cHu', 'c3W']

storage = StorageConfiguration(
    output=[
        "hdfs://eddie.crc.nd.edu:19000/store/user/$USER/ttV/{}/{}/".format(gridpack_version, gen_version),
        "file:///hadoop/store/user/$USER/ttV/{}/{}/".format(gridpack_version, gen_version),
        # "root://deepthought.crc.nd.edu//store/user/$USER/ttV/{}/{}".format(gridpack_version, gen_version),
        "chirp://eddie.crc.nd.edu:9094/store/user/$USER/ttV/{}/{}/".format(gridpack_version, gen_version),
        "gsiftp://T3_US_NotreDame/store/user/$USER/ttV/{}/{}/".format(gridpack_version, gen_version),
        "srm://T3_US_NotreDame/store/user/$USER/ttV/{}/{}/".format(gridpack_version, gen_version)
    ],
    input=[
        # "hdfs:///store/user/$USER/ttV/" + gridpack_version,
        # "file:///hadoop/store/user/$USER/ttV/" + gridpack_version,
        "root://deepthought.crc.nd.edu//store/user/$USER/ttV/{}/".format(gridpack_version),
        # "chirp://eddie.crc.nd.edu:9094/store/user/$USER/ttV/" + gridpack_version,
        # "gsiftp://T3_US_NotreDame/",
        # "srm://T3_US_NotreDame/"
    ],
    disable_input_streaming=True
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
for process in ['ttW', 'ttZ', 'ttH']:
    for operator in operators:
        lhe = Workflow(
                label='{}_lhe_{}'.format(process, operator),
                # command='source /opt/crc/Modules/current/init/sh; module unload gcc; module load gcc/5.2.0; cmsRun',
                pset='HIG-RunIIWinter15wmLHE-01035_1_cfg.py',
                sandbox=cmssw.Sandbox(release=release),
                outputs=['HIG-RunIIWinter15wmLHE-01035ND.root'],
                dataset=MultiProductionDataset(
                    gridpacks='{}_gridpacks_{}'.format(process, operator),
                    events_per_gridpack=2000,
                    events_per_task=2000
                ),
                category=lhe_resources,
        )

        gen = Workflow(
                label='{}_gen_{}'.format(process, operator),
                pset='HIG-RunIISummer15Gen-01168_1_cfg.py',
                sandbox=cmssw.Sandbox(release=release),
                merge_size='3.5G',
                cleanup_input=True,
                outputs=['HIG-RunIISummer15Gen-01168ND.root'],
                dataset=ParentDataset(
                    parent=lhe,
                    units_per_task=1
                ),
                category=gen_resources
        )

        workflows.extend([lhe, gen])

config = Config(
    label='ttV.{}.{}.gen'.format(gridpack_version, gen_version),
    workdir='/tmpscratch/users/$USER/ttV/{}/{}/'.format(gridpack_version, gen_version),
    plotdir='~/www/lobster/ttV/{}/{}/'.format(gridpack_version, gen_version),
    storage=storage,
    workflows=workflows,
    advanced=AdvancedOptions(log_level=1, bad_exit_codes=[127], use_dashboard=False),
    # elk=ElkInterface('elk.crc.nd.edu', 9200, 'elk.crc.nd.edu', 5601, project='ttV.{}.{}'.format(gridpack_version, gen_version), dashboards=['Core', 'Advanced', 'Tasks'])
)
