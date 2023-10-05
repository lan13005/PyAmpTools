import os
import pytest


@pytest.mark.amppars
def test_AmpPars():
    REPO_HOME = os.environ["REPO_HOME"]
    base_dir = f'{REPO_HOME}/tests/test_files'
    fit_results = f'{base_dir}/result.fit'
    cmd = f'python {base_dir}/AmpPars.py {fit_results}'
    print(cmd)
    os.system(cmd)
