import glob
import inspect
import os
import re
import subprocess
import unittest

import psutil
from IPython.display import Code
import math


def calculate_subplot_grid_size(length_of_list):
    sqrt_length = math.sqrt(length_of_list)
    rows = math.floor(sqrt_length)
    columns = rows

    while rows * columns < length_of_list:
        if columns == rows: columns += 1
        else: rows += 1

    return rows, columns

def PrintSourceCode(function):
    """Returns the source code of a function"""
    src_code = "".join(inspect.getsourcelines(function)[0])
    return Code(src_code, language="python")


def get_function_by_pid(pid):
    """Returns the function or executable associated with a process ID"""
    try:
        process = psutil.Process(pid)
        function = (
            process.name()
        )  # This retrieves the name of the function or executable associated with the process.
        return function
    except psutil.NoSuchProcess:
        return "Process with PID {} not found.".format(pid)
    except psutil.AccessDenied:
        return "Access denied to process with PID {}.".format(pid)


def get_pid_family():
    """
    Returns the function or executable associated with the current process ID and its parent process ID

    Example:
        1) python example.py              -> python, bash
        2) mpirun -np 1 python example.py -> python, mpirun
    """
    pid = os.getpid()
    ppid = os.getppid()
    current = get_function_by_pid(pid)
    parent = get_function_by_pid(ppid)
    return current, parent


def check_nvidia_devices():
    """Checks nvidia-smi for devices"""
    try:
        subprocess.check_output(["nvidia-smi"])
        return True, "NVIDIA devices exist on your system."
    except Exception:
        return False, "No NVIDIA devices found."


def remove_all_whitespace(string):
    """Removes all whitespace from a string"""
    return string.replace(" ", "")


class testKnownFailure(unittest.TestCase):
    """
    Function wrapper that tests for known failures.
    Only accepts args
    """

    def __init__(self, function):
        super().__init__()
        self.function = function

    def __call__(self, *args):
        str_args = ", ".join([str(arg) for arg in args])
        str_func = f"{self.function.__name__}({str_args})"
        statement = f"Testing known failure of function: \n   {str_func} "
        print(f"{statement:.<100} ", end="")
        with self.assertRaises(AssertionError) as context:
            self.function(*args)
        print("Passed!")


def print_dict(d):
    for k, v in d.items():
        print(f"{k}: {v}")


def check_shared_lib_exists(libName, verbose=False):
    """Check if a shared library exists in the LD_LIBRARY_PATH"""
    assert "." in libName, "libName must include file extension"

    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    libExists = False
    for path in ld_library_path.split(":"):
        searchPath = os.path.join(path, f"{libName}")
        if verbose:
            print(f"Searching for {libName} in {searchPath}", end="")
        if os.path.exists(searchPath):
            libExists = True
            if verbose:
                print(" --- Found here!")
            break
        if verbose:
            print()

    return libExists


def raiseError(errorType, msg):
    """Raise an error of type errorType with message msg"""
    raise errorType(msg)


def get_captured_number(filename, captured_loc, prefix, suffix):
    """
    Extracts the numeric part from a filename based on a specific pattern.

    Args:
        filename (str): The file name from which to extract the number.

    Returns:
        int: The extracted number.
    """
    # Assuming the number is located between the last underscore and the file extension

    _filename = filename.split("/")[captured_loc]

    match = re.search(rf"{prefix}(\d+){suffix}", _filename)
    if match:
        return int(match.group(1))
    else:
        # Return a default value if no number is found, to avoid sorting errors
        return float("inf")


def glob_sort_captured(files):
    """
    glob files and sort based on captured number. [] denote the captured location.
    Example: files = 'FOLDER*/err_[].log'
        {FOLDER1/err_2.log, FOLDER3/err_1.log} should return {FOLDER3/err_1.log, FOLDER1/err_2.log} as 2, 1 are [] captures

    Args:
        files (str): glob pattern with [] denoting the capture location

    Returns:
        list: sorted list of files based on captured number
    """

    # return input if no sorting is necessary
    if "[]" not in files or "*" not in files:
        return [files]

    files = files.rstrip("/")  # Remove right trailing slashes

    # Find location of capture []
    fname = files.split("/")
    capture_loc = [i for i, f in enumerate(fname) if "[]" in f]
    assert len(capture_loc) == 1, "Only one capture location allowed"
    captured_loc = capture_loc[0]
    # Get prefix and suffix for regex use
    prefix, suffix = fname[captured_loc].split("[]")

    if "[]" in files:
        files = files.replace("[]", "*")
        files = glob.glob(files)
        files = sorted(
            files,
            key=lambda fname: get_captured_number(fname, captured_loc, prefix, suffix),
        )
    elif "[]" not in files and "*" in files:
        files = glob.glob(files)
        files = sorted(
            files,
            key=lambda fname: get_captured_number(fname, captured_loc, prefix, suffix),
        )

    return files


def safe_getsize(file_path):
    """Return file size if file exists, else return 0"""
    try:
        return os.path.getsize(file_path)
    except FileNotFoundError:
        return 0


###############################################################################################
def setPlotStyle(
    small_size: int = 14,
    big_size: int = 16,
) -> None:

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import mplhep as hep

    """Set the plotting style."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use([hep.styles.ATLAS])
    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=big_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small_size)  # legend fontsize
    plt.rc("figure", titlesize=big_size)  # fontsize of the figure title


###############################################################################################
