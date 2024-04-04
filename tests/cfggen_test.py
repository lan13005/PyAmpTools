import os
import pytest

def test_cfgGenerator():
   REPO_HOME = os.environ["REPO_HOME"]
   SCRIPTS = f'{REPO_HOME}/scripts'
   cmd = [f'python {SCRIPTS}/cfggen.py']
   output_file = 'EtaPi.cfg'
   cmd = ' '.join(cmd)
   print(cmd)
   os.system(cmd)
   assert( os.path.exists(output_file) and os.path.getsize(output_file) > 0 )
   os.system(f'rm {output_file}')
