import os
import subprocess
import pytest

@pytest.mark.fit
def test_fit():
	REPO_HOME = os.environ['REPO_HOME']
	cmd=f"python {REPO_HOME}/EXAMPLES/python/fit.py {REPO_HOME}/tests/samples/fit_res.cfg"
	print(cmd)
	return_code = subprocess.call(cmd, shell=True)
	print(return_code)
	os.system(r'rm -f *.fit normint*') # clean up
	assert return_code == 0, f"Command '{cmd}' returned a non-zero exit code: {return_code}"
