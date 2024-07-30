import glob
import os
import subprocess
import sys
from os.path import expandvars
from shutil import which


def recursive_link_if_not_exist(source_folders, destination_folder, ftype, verbose=True):
    if verbose:
        print(f"\n\nDestination Folder: {destination_folder}")
    for source_folder in source_folders:
        for source in glob.glob(source_folder + "/" + ftype):
            if source == sys.argv[0]:
                continue  # do not add this linker itself...
            destination_file = destination_folder + "/" + source.split("/")[-1]
            cmd = f"ln -snfr {source} {destination_file}"
            if verbose:
                print(f"Re-Linking:\n   SRC:  {source}\n   DEST: {destination_file}")
            os.system(cmd)
    if verbose:
        print("-- Added above modules to conda environment... ---")


PYAMPTOOLS_HOME = expandvars("$PYAMPTOOLS_HOME")

source_folders = [
    f"{PYAMPTOOLS_HOME}/utility/",
]

####################################
## Link additional C hearders to PyROOTs include directory
####################################
# Retrieve the output of this bash command "root-config --incdir"
#   which gives the include directory for PyROOT
cmd = "root-config --incdir"
# check if root-config in path
if which("root-config"):
    incdir = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    recursive_link_if_not_exist(source_folders, incdir, "*.h")
else:
    print("root-config not found in path. Not linking *.h files to root's include directory.")
