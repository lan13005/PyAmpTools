import os
import pytest

@pytest.mark.plotgenrdf
def test_plotgen():
   REPO_HOME = os.environ["REPO_HOME"]
   fit_results = f'{REPO_HOME}/tests/test_files/result.fit'
   output_file = f'{REPO_HOME}/tests/plotgenrdf_test' # leave ftype as program appends .pdf type
   cmd = [f'python {REPO_HOME}/EXAMPLES/python/PlotGenRDF.py', fit_results, '-o', output_file]
   cmd = ' '.join(cmd)
   print(cmd)
   os.system(cmd)
   assert( os.path.exists(f'{output_file}.pdf') ), f'PlotGenRDF.py failed to generate {output_file}.pdf'
   os.system(f'rm {output_file}.pdf')
