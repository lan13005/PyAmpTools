import os
import subprocess
import pytest
import re

pmp = pytest.mark.parametrize

REPO_HOME = os.environ['REPO_HOME']

def fit(accelerator, cfgfile):
	cmd=f"pa fit {cfgfile} --accelerator '{accelerator}'"
	print(cmd)
	output = subprocess.check_output(cmd, shell=True)
	output = output.decode('utf-8')
	print(output)
	assert( os.path.exists('result_0.fit') and os.path.exists('seed_0.txt') ), "MLE fit failed to produce result_0.fit and/or seed_0.txt file"
	os.system(r'rm -f *.fit normint* *seed*') # clean up
	# use regex search output for a line that starts with Final Likelihood and extract the number at the end
	nll = float(re.search(r'Final Likelihood: (\d+\.?\d*)', output).group(1))
	assert( abs(nll-14346.40812) < 1e-5 ), f"nll = |{nll}-14346.40812| > 1e-5"

## The below tests might come back and bite me.
# gpu and mpi will be accessed only if available and will fall back
# to cpu if unavailable. These tests can be positive even if
# mpi or gpu is not tested

@pmp("args", [ ("cpu", f"{REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/fit.cfg"), 
               ("gpu", f"{REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/fit.cfg"), ] )
def test_fit_cpu(args):
    accelerator, cfgfile = args
    print(f"testing fit with accelerator: {accelerator}")
    fit(accelerator, cfgfile)
