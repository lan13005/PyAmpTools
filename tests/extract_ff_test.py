import os
import pytest

@pytest.mark.extract_ff
def test_extract_ff():
   REPO_HOME = os.environ["REPO_HOME"]
   fit_results = f'{REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/result.fit'
   output_file = f'{REPO_HOME}/tests/ff.txt'
   cmd = [f'python {REPO_HOME}/EXAMPLES/python/extract_ff.py', fit_results, output_file, '-regex_merge', "'.*::(.*)::.*~>\\1'", "'.*(.)$~>\\1'"]
   cmd = ' '.join(cmd)
   print(cmd)
   os.system(cmd)
   assert( os.path.exists(output_file) and os.path.getsize(output_file) > 0 )
   os.system(f'rm {output_file}')
