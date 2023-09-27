import sys
import os
from os.path import expandvars, exists
import glob

###################################
# Manually adding some libraries to conda's virtual environment
#   instead of exporting globally to pythonpath
###################################

python_version = sys.version
major_minor_versions = '.'.join(python_version.split('.')[:2])

destination = f"$CONDA_PREFIX/lib/python{major_minor_versions}/site-packages"

source_folders = [
        f"{os.getcwd()}/utils/",
        ]

for source_folder in source_folders:
  for source in glob.glob(source_folder+'*.py'):
    if source == sys.argv[0]: continue # do not add this linker itself...
    cmd = f'ln -nsf {source} {destination}'
    print(cmd)
    os.system(cmd)
print("Added above modules to conda environment...")
