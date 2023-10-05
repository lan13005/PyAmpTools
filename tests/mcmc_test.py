import os
import subprocess
import pytest

@pytest.mark.mcmc
def test_mcmc():
   REPO_HOME = os.environ['REPO_HOME']
   cfgfile = f'{REPO_HOME}/gen_amp/fit_res.cfg'
   mle_fit = f'{REPO_HOME}/tests/test_files/result.fit'
   ofolder = f'{REPO_HOME}/tests/mcmc'
   cmd=f"python {REPO_HOME}/EXAMPLES/python/mcmc.py {cfgfile} {mle_fit} -o {ofolder} -f 'mcmc.h5' -n 20 -b 10 -s 10 -overwrite"
   print(cmd)
   return_code = subprocess.call(cmd, shell=True)
   print(return_code)
   assert return_code == 0, f"Command '{cmd}' returned a non-zero exit code: {return_code}"
   subprocess.call(f"rm -rf {ofolder}", shell=True) # clean up
   assert not os.path.exists(ofolder), f"Folder {ofolder} was not deleted!"
