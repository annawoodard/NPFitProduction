#!/bin/sh

cat <<EOF
=========================================================================
this script creates a working directory for EffectiveTTV production
output is in setup.log
=========================================================================
EOF

(
set -e
set -o xtrace

export SCRAM_ARCH=slc6_amd64_gcc491
scramv1 project CMSSW_7_4_7
cd CMSSW_7_4_7/src
set +o xtrace
eval $(scramv1 runtime -sh)
set -o xtrace
git cms-init > /dev/null

git clone git@github.com:annawoodard/EffectiveTTVProduction.git EffectiveTTVProduction/EffectiveTTVProduction

scram b -j 32

) > setup.log

cat <<EOF
=========================================================================
output is in setup.log
=========================================================================
EOF
