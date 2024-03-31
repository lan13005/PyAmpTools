import os
import subprocess
import pytest

REPO_HOME = os.environ['REPO_HOME']
cfgfile = f'{REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/fit.cfg'
mle_fit = f'{REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/result.fit'
ofolder = f'{REPO_HOME}/tests/mcmc'

@pytest.mark.mcmc
def test_mcmc():
   cmd=f"pa mcmc {cfgfile} \
            --ofile '{ofolder}/emcee_state.h5' \
            --nwalkers 20 \
            --burnin 10 \
            --nsamples 100 \
            --corner_ofile '{ofolder}/corner.png' \
            --intensity_dump '{ofolder}/samples_intensity.feather' \
            --overwrite"
   print(cmd)
   return_code = subprocess.call(cmd, shell=True)
   print(return_code)
   assert return_code == 0, f"Command '{cmd}' returned a non-zero exit code: {return_code}"
   # subprocess.call(f"rm -rf {ofolder}", shell=True) # clean up
   # assert not os.path.exists(ofolder), f"Folder {ofolder} was not deleted!"
