import os
import pytest

@pytest.mark.plotgen
def test_plotgen():
   REPO_HOME = os.environ["REPO_HOME"]
   fit_results = f'{REPO_HOME}/tests/test_files/result.fit'
   output_root_file = f'{REPO_HOME}/tests/plotgen_test.root'
   cmd = [f'python {REPO_HOME}/EXAMPLES/python/PlotGen.py', fit_results, '-o', output_root_file]
   cmd = ' '.join(cmd)
   print(cmd)
   os.system(cmd)
   assert( os.path.exists(output_root_file) and os.path.getsize(output_root_file) > 1000 )
   os.system(f'rm {output_root_file}')
