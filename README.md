This code calculates cross sections and produces events using a Madgraph model file describing the new physics (NP), scanning the requested phase space of Wilson coefficients. Multidimensional scans (changing multiple coefficients per scan) is supported. Currently only the [Higgs Effective Lagrangian](https://arxiv.org/abs/1310.5150) model is implemented. The code relies on [Lobster](https://github.com/matz-e/lobster) for workflow management. In the following instructions a general familiarity with Madgraph and Lobster is assumed-- see the [Madgraph documentation](https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/ManualAndHelp) and [Lobster documentation](http://lobster.readthedocs.io/en/latest/).
## Installation
To install Lobster, see the [documentation](http://lobster.readthedocs.io/en/latest/). Next, set up your working directory:

    curl https://raw.githubusercontent.com/annawoodard/EffectiveTTVProduction/master/setup.sh|sh -

## Quick start
### Scanning cross sections
When scanning Wilson coefficients, it is not usually obvious what range of values are optimal.  The approach used here is to start with a large range and zoom in until `(NP cross section) / (SM cross section) < scale` for a given set of processes, where `scale` is set by the user. To scan cross sections, edit [test/cross_sections.py](test/cross_sections.py). This Lobster configuration should work out of the box without modification for testing, but for your own purposes you will need to adjust the variables defined at the top. The following commands assume you are in the `test` directory. To run, activate your Lobster virtualenv:

    . ~/.lobster/bin/activate

Next, start the Lobster processing run:

    lobster process cross_sections.py

Finally, start a work queue factory to submit your jobs. If you are running at Notre Dame, login to a CRC headnode (for example `condorfe.crc.nd.edu`) and execute:

    nohup work_queue_factory -T condor -M lobster_$USER.*ttV.* -d all -o /tmp/${USER}_lobster_ttV_factory.debug -C $(readlink -f factory_cross_sections_CRC.json) --wrapper "python /afs/crc.nd.edu/group/ccl/software/runos/runos.py rhel6" --extra-options="--workdir=/disk" --worker-binary=/afs/crc.nd.edu/group/ccl/software/x86_64/redhat6/cctools/$cctools/bin/work_queue_worker >&  /tmp/${USER}_lobster_ttV.log &

Submitting to the CRC cluster is recommended as there are far more resources. For additional resources, you can submit to the ND T3 by logging into to `earth.crc.nd.edu` and executing:

    nohup work_queue_factory -T condor -M lobster_$USER.*ttV.* -d all -o /tmp/${USER}_lobster_ttV_factory.debug -C $(readlink -f factory_cross_sections_T3.json) >& /tmp/${USER}_lobster_ttV_xsec.log &
    
If you are not running at Notre Dame, you will need to setup [cctools](https://ccl.cse.nd.edu/software/) yourself. It should be possible to submit jobs by substituting `-T condor` in the factory command above with your batch system type. Supported batch systems include condor, sge, torque, moab, slurm, chirp, and amazon.


### Producing gen samples
The Lobster configuration for producing gen samples is located at [test/gen.py](test/gen.py). Two approaches to finding the range of Wilson coefficient values are supported. A `low` and `high` interval can be specified, which will fix the range to [low, high] for each coefficient. Alternatively, a scan from the previous step can be used to restrict the ranges such that `(NP cross section) / (SM cross section) < scale`. The configuration runs the interval method out of the box without modification for testing, but for your own purposes you will need to adjust the variables defined at the top. The following commands assume you are in the `test` directory. To run, activate your Lobster virtualenv:

    . ~/.lobster/bin/activate
    
Next, start the Lobster processing run:

    lobster process gen.py

Finally, start a work queue factory to submit your jobs (currently only submitting from the ND T3 or other cluster with access to `/cvmfs` is supported):

    nohup work_queue_factory -T condor -M lobster_$USER.*ttV.*gen -d all -o /tmp/${USER}_lobster_ttV_gen.debug -C $(readlink -f factory_gen_T3.json) >& /tmp/${USER}_lobster_ttV_gen.log &

## More details
### Producing gen samples using the scaling method
Run a cross section scan. Then merge the output into a single file (replace the directory below with the output directory of the scan):

    merge_scans scans.npz /hadoop/store/user/$USER/ttV/cross_sections/1/final_scan_*/*npz

Now replace the `command` in [test/gen.py](test/gen.py) with the commented out one and add `scans.npz` to `extra_inputs`, and follow the instructions above for producing gen samples.

### Multidimensional scans
For multidimensional scans increase `dimension` in your Lobster configuration file. Each coefficient can take on `numvalues` different values, which are chosen in evenly spaced intervals between `low` and `high`. As an example, consider a Lobster configuration file with the following definitions:

    coefficients = ['cuB', 'cuW']
    constraints = ['ttH', 'ttZ', 'ttW']
    dimensions = 2
    low = -2
    high = 2
    numvalues = 3

Both `cuB` and `cuW` will be sampled at values -2.,  0.,  and 2. The sampled points will be (the first column corresponds to `cuB` and the second to `cuW`):

    array([[-2., -2.],
           [-2.,  0.],
           [-2.,  2.],
           [ 0., -2.],
           [ 0.,  0.],
           [ 0.,  2.],
           [ 2., -2.],
           [ 2.,  0.],
           [ 2.,  2.]])
           
When using the scaling method of coefficient rangefinding, an input 'coarse-grained scan' is requred. All points for which none of the `constraint` processes exceed `(NP cross section) / (SM cross section) < scale` will be found. From those points, the maximum and minimum value for each coefficient is set as `low` and `high` for that coefficient. Note that due to interference effects between coefficients, a more sophisticated approach might be needed.

### Using the gen samples
By default [test/gen.py](test/gen.py) produces one output dataset per group of coefficients to scan, lumping all parameter points together. The sampled parameter point is saved to each event for further processing; see [plugins/wilsonCoefficientAnnotator.cc](plugins/wilsonCoefficientAnnotator.cc).

### Adding processing steps
Lobster can chain together processing steps. If you want to run reco, etc on your gen samples, create a new workflow in your Lobster configuration file by copying the `gen` workflow and adjusting the `pset` and `outputs` to match the CMSSW pset that runs your subsequent step.
