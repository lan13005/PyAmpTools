import sys
import os
from os.path import expandvars, exists
import glob
import subprocess
from shutil import which

def recursive_link_if_not_exist(source_folders, destination_folder, ftype, verbose=False):
  if verbose: print(f"\n\nDestination Folder: {destination_folder}")
  for source_folder in source_folders:
    for source in glob.glob(source_folder+'/'+ftype):
      if source == sys.argv[0]: continue # do not add this linker itself...
      destination_file = destination_folder + '/' + source.split('/')[-1]
      cmd = f'ln -snfr {source} {destination_file}'
      if verbose: print(f'Re-Linking:\n   SRC:  {source}\n   DEST: {destination_file}')
      os.system(cmd)
  if verbose: print("-- Added above modules to conda environment... ---")

REPO_HOME = expandvars("$REPO_HOME")

source_folders = [
        f"{REPO_HOME}/utility/",
        f"{REPO_HOME}/EXAMPLES/python",
        ]

###################################
# Manually adding some libraries to conda's virtual environment
#   instead of exporting globally to pythonpath
###################################
python_version = sys.version
major_minor_versions = '.'.join(python_version.split('.')[:2])

destination_folder = f"$CONDA_PREFIX/lib/python{major_minor_versions}/site-packages"
destination_folder = expandvars(destination_folder)

recursive_link_if_not_exist(source_folders, destination_folder, '*.py')

####################################
## Link additional C hearders to PyROOTs include directory
####################################
# Retrieve the output of this bash command "root-config --incdir"
#   which gives the include directory for PyROOT
cmd = "root-config --incdir"
# check if root-config in path
if which("root-config"):
  incdir = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
  recursive_link_if_not_exist(source_folders, incdir, '*.h')
else:
  print("root-config not found in path. Not linking *.h files to root's include directory.")
