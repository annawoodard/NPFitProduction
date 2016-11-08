# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Configuration/GenProduction/python/HIG-RunIIWinter15wmLHE-01035-fragment.py --fileout file:HIG-RunIIWinter15wmLHE-01035ND.root --mc --eventcontent LHE --datatier LHE --conditions MCRUN2_71_V1::All --step LHE --python_filename HIG-RunIIWinter15wmLHE-01035_1_cfg.py --no_exec --customise Configuration/DataProcessing/Utils.addMonitoring -n 2320
import json
import subprocess

import FWCore.ParameterSet.Config as cms

process = cms.Process('LHE')

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(2320)
)

process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet()

process.configurationMetadata = cms.untracked.PSet(
    version=cms.untracked.string('$Revision: 1.19 $'),
    annotation=cms.untracked.string('Configuration/GenProduction/python/HIG-RunIIWinter15wmLHE-01035-fragment.py nevts:2320'),
    name=cms.untracked.string('Applications')
)

process.LHEoutput = cms.OutputModule("PoolOutputModule",
    splitLevel=cms.untracked.int32(0),
    eventAutoFlushCompressedSize=cms.untracked.int32(5242880),
    outputCommands=process.LHEEventContent.outputCommands,
    fileName=cms.untracked.string('file:HIG-RunIIWinter15wmLHE-01035ND.root'),
    dataset=cms.untracked.PSet(
        filterName=cms.untracked.string(''),
        dataTier=cms.untracked.string('LHE')
    )
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'MCRUN2_71_V1::All', '')

process.externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    nEvents = cms.untracked.uint32(2320),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh'),
    numberOfParameters = cms.uint32(1),
    args = cms.vstring('/hadoop/store/user/awoodard/ttV/3/ttW_gridpacks_cuG/gridpack.tar_5716.xz')
)

with open('parameters.json', 'r') as f:
    config = json.load(f)

import os
subprocess.call(['tar', 'xaf', os.path.basename(config['mask']['files'][0].replace('file:', '')), 'point.json'])
with open('point.json', 'r') as f:
    info = json.load(f)

process.annotator = cms.EDProducer('WilsonCoefficientAnnotator',
        wilsonCoefficients = cms.vdouble(*info['coefficients']),
        operators = cms.vstring(*[str(x) for x in info['operators']]),
        point=cms.int32(info['point'])

)

process.LHEoutput.outputCommands += ['keep *_annotator_*_*']

process.lhe_step = cms.Path(process.externalLHEProducer*process.annotator)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.LHEoutput_step = cms.EndPath(process.LHEoutput)

process.schedule = cms.Schedule(process.lhe_step,process.endjob_step,process.LHEoutput_step)

from Configuration.DataProcessing.Utils import addMonitoring 
process = addMonitoring(process)

