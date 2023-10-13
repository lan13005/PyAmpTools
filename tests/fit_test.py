import os
import subprocess
import pytest

def fit(accelerator):
	REPO_HOME = os.environ['REPO_HOME']
	cmd=f"python {REPO_HOME}/EXAMPLES/python/fit.py {REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/fit.cfg"
	print(cmd)
	output = subprocess.check_output(cmd, shell=True)
	os.system(r'rm -f *.fit normint*') # clean up
	nll = float(output.split()[-1])
	assert( abs(nll-14430.0412) < 1e-4 ), f"nll = {nll} !~= 14430.0412"


@pytest.mark.fit
def test_fit_cpu():
	fit("")

## The below tests might come back and bite me
# gpu and mpi will be accessed only if available and will fall back
# to cpu if unavailable. These tests can be positive even if
# mpi or gpu is not tested

@pytest.mark.fit
def test_fit_gpu():
	fit("gpu")

@pytest.mark.fit
def test_fit_mpi():
	fit("MPI")

@pytest.mark.fit
def test_fit_mpigpu():
	fit("MPIGPU")
