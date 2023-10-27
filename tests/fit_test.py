import os
import subprocess
import pytest
import re

def fit(accelerator):
	REPO_HOME = os.environ['REPO_HOME']
	cmd=f"python {REPO_HOME}/EXAMPLES/python/fit.py {REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/fit.cfg --accelerator '{accelerator}'"
	print(cmd)
	output = subprocess.check_output(cmd, shell=True)
	output = output.decode('utf-8')
	print(output)
	os.system(r'rm -f *.fit normint*') # clean up
	# use regex search output for a line that starts with Final Likelihood and extract the number at the end
	nll = float(re.search(r'Final Likelihood: (\d+\.?\d*)', output).group(1))
	assert( abs(nll-14346.40812) < 1e-5 ), f"nll = |{nll}-14346.40812| > 1e-5"


@pytest.mark.fit
def test_fit_cpu():
	fit("")

## The below tests might come back and bite me.
# gpu and mpi will be accessed only if available and will fall back
# to cpu if unavailable. These tests can be positive even if
# mpi or gpu is not tested

@pytest.mark.fit
def test_fit_gpu():
	fit("gpu")

# @pytest.mark.fit
# def test_fit_mpi():
# 	fit("mpi")

# @pytest.mark.fit
# def test_fit_mpigpu():
# 	fit("mpigpu")
