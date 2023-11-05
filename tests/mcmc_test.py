import os
import subprocess
import pytest

@pytest.mark.mcmc
def test_mcmc():
   REPO_HOME = os.environ['REPO_HOME']
   cfgfile = f'{REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/fit.cfg'
   mle_fit = f'{REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/result.fit'
   ofolder = f'{REPO_HOME}/tests/mcmc'
   cmd=f"python {REPO_HOME}/EXAMPLES/python/mcmc.py --cfgfile {cfgfile} --ofile f'{ofolder}/mcmc.h5' \
            --nwalkers 20 \
            --burnin 10 \
            --nsamples 100 \
            --overwrite"
   print(cmd)
   return_code = subprocess.call(cmd, shell=True)
   print(return_code)
   assert return_code == 0, f"Command '{cmd}' returned a non-zero exit code: {return_code}"
   subprocess.call(f"rm -rf {ofolder}", shell=True) # clean up
   assert not os.path.exists(ofolder), f"Folder {ofolder} was not deleted!"
