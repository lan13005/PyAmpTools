import os
import pytest

@pytest.mark.cfggenerator
def test_cfgGenerator():
   REPO_HOME = os.environ["REPO_HOME"]
   cmd = [f'python {REPO_HOME}/EXAMPLES/python/cfgGenerator.py']
   output_file = 'EtaPi.cfg'
   cmd = ' '.join(cmd)
   print(cmd)
   os.system(cmd)
   assert( os.path.exists(output_file) and os.path.getsize(output_file) > 0 )
   os.system(f'rm {output_file}')
