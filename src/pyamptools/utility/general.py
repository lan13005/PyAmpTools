import glob
import inspect
import math
import os
import re
import subprocess
import time
import unittest

import psutil
import yaml
from IPython.display import Code
from omegaconf import OmegaConf
from omegaconf._utils import OmegaConfDumper, _ensure_container

# ===================================================================================================
# ===================================================================================================

spectroscopic_map = {0: "S", 1: "P", 2: "D", 3: "F"}
lmax = max(spectroscopic_map.keys())


def zlm_amp_name(refl, l, m):
    if l < 0:
        raise ValueError("zlm_amp_name: l must be greater than or equal to 0!")
    if l > lmax:
        raise ValueError(f"zlm_amp_name: l must be less than or equal to {lmax}!")
    if m < -l or m > l:
        raise ValueError("zlm_amp_name: m must be between -l and l!")
    m_sign = "p" if m >= 0 else "m"
    _refl = "+" if refl == 1 else "-"
    _m = f"{m_sign}{abs(m)}"
    return f"{spectroscopic_map[l]}{_m}{_refl}"


def vps_amp_name(refl, J, M, l):
    if J < 0:
        raise ValueError("vps_amp_name: J must be greater than or equal to 0!")
    if M < -J or M > J:
        raise ValueError("vps_amp_name: M must be between -J and J!")
    if l < abs(J) - 1:
        raise ValueError("vps_amp_name: l must be greater than or equal to 0!")
    if l > lmax:
        raise ValueError(f"vps_amp_name: l must be less than or equal to {lmax}!")
    M_sign = "p" if M >= 0 else "m"
    _refl = "+" if refl == 1 else "-"
    return f"{J}{spectroscopic_map[l]}{M_sign}{abs(M)}{_refl}"


converter = {}  # i.e. {'Sp0+': [1, 0, 0]}
prettyLabels = {}  # i.e. {'Sp0+': '$S_{0}^{+}$'}
example_zlm_names = []  # i.e. ['Sp0+', 'Pp1+', 'Dm2-']
example_vps_names = []

refls = ["-", "+"]
for e, refl in zip([-1, 1], refls):
    for l in range(lmax + 1):
        ### For Zlm two pseudoscalar waves
        for m in range(-l, l + 1):
            amp = zlm_amp_name(e, l, m)
            example_zlm_names.append(amp)
            converter[amp] = [e, l, m]
            prettyLabels[amp] = rf"${spectroscopic_map[l]}_{{{m}}}^{{{refl}}}$"

        ### For Vps two pseudoscalar waves
        for J in range(abs(1 - l), abs(1 + l + 1)):
            for M in range(-J, J + 1):
                amp = vps_amp_name(e, J, M, l)
                example_vps_names.append(amp)
                converter[amp] = [e, l, M, J]
                P = "+" if (-1) ** l == +1 else "-"
                C = "-"  # vector-pseudoscalar is always -1
                prettyLabels[amp] = rf"${J}^{{{P}{C}}}[{spectroscopic_map[l]}_{{{M}}}^{{{refl}}}]$"

# If there is no prettyLabel version, just pass back the key
class KeyReturningDict(dict):
    def __missing__(self, key):
        return key
prettyLabels = KeyReturningDict(prettyLabels)

# ===================================================================================================
# ===================================================================================================


def load_yaml(path_to_yaml, resolve=True):
    """
    Load a yaml file into a dictionary. If default_yaml field exists, overwrite default yaml with current yaml

    Args:
        path_to_yaml (str): Path to the yaml file
        resolve (bool): Whether to resolve (variable interpolation) the yaml file
    """
    try:
        yaml = OmegaConf.load(path_to_yaml)
        yaml = OmegaConf.to_container(yaml, resolve=resolve)
        if "default_yaml" in yaml:
            default = OmegaConf.load(yaml["default_yaml"])
            yaml = OmegaConf.merge(default, yaml)  # Merge working yaml INTO the default yaml, ORDER MATTERS!
        return yaml
    except Exception as e:
        print(f"Error loading yaml file: {e}")
        return None


class YamlDumper(OmegaConfDumper):
    """
    yaml.dump accepts a custom dumper class. We keep lists (for leaf nodes) in flow style, i.e. : [1, 2, 3] instead of multi-line indented dashes
    """

    def represent_sequence(self, tag, sequence, flow_style=False):
        # Check if the current node being represented is a list
        if isinstance(sequence, list):
            return super().represent_sequence(tag, sequence, flow_style=True)
        else:
            return super().represent_sequence(tag, sequence, flow_style=flow_style)


def get_yaml_dumper():
    # Taken from omegaconf._utils.py
    if not YamlDumper.str_representer_added:
        YamlDumper.add_representer(str, YamlDumper.str_representer)
        YamlDumper.str_representer_added = True
    return YamlDumper


def dump_yaml(cfg, output_file_path, indent=4, resolve=False):
    """
    Function enforcing flow style for lists

    Parameters
    ----------
    cfg : OmegaConf / dict
        Configuration object to be dumped. OmegaConf or dict that can be converted
    output_file_path : str
        Path to dump the configuration file to
    indent : int, optional
        Indentation (how many spaces per level), by default 4
    resolve : bool, optional
        Resolve variables (intepolation) in cfg before dumping, by default False
    """
    if not isinstance(cfg, OmegaConf):
        try:
            cfg = OmegaConf.create(cfg)
        except Exception:
            raise ValueError("cfg could not be converted to OmegaConf object")
    _ensure_container(cfg)
    container = OmegaConf.to_container(cfg, resolve=resolve, enum_to_str=True)
    with open(output_file_path, "w") as f:
        yaml.dump(  # type: ignore
            container,
            f,
            indent=indent,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            Dumper=get_yaml_dumper(),
        )


class Timer:
    def __init__(self):
        """No reason not to start the timer upon creation right?"""
        self._start_time = time.time()

    def read(self):
        """Read the start time, stop time, and elapsed time."""

        read_time = time.time()
        elapsed_time = read_time - self._start_time

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._start_time))
        read_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(read_time))
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        return start_time, read_time, elapsed_time


def get_gpu_status():
    from subprocess import getoutput

    cmd = "scontrol show node sciml1902,sciml1903,sciml2101,sciml2102,sciml2103,sciml2301,sciml2302"
    out = getoutput(cmd)

    nodes = out.split("\n\n")

    results = []

    def try_selection(line, delim1, id1, delim2, id2, type_cast):
        try:
            output = type_cast(line.split(delim1)[id1].split(delim2)[id2])
        except Exception:
            output = 0  # AllocTres empty means no GPUs allocated
        return output

    for node in nodes:
        node = node.split("\n")
        for line in node:
            if "NodeName" in line:
                node_name = try_selection(line, "=", 1, " ", 0, str)
            if "CPUAlloc" in line:
                cpu_alloc = try_selection(line, "=", 1, " ", 0, int)
                cpu_total = try_selection(line, "=", 2, " ", 0, int)
            if "Gres" in line:
                gres = try_selection(line, "=", 1, " ", 0, str)
            if "RealMemory" in line:
                free_mem = int(try_selection(line, "=", 3, " ", 0, float) / 1000)
                tot_mem = int(try_selection(line, "=", 1, " ", 0, float) / 1000)
            if "CfgTRES" in line:
                configured_gpus = try_selection(line, ",", -1, "=", 1, int)
            if "AllocTRES" in line:
                allocated_gpus = try_selection(line, ",", -2, "=", 1, int)

        output = f"{node_name}" if node_name is not None else "N/A"
        results.append(f"\n{'Node Name':<40}: {output}")

        output = f"{cpu_alloc}/{cpu_total} OR {(cpu_total-cpu_alloc)/cpu_total*100:.2f}% free" if cpu_alloc is not None and cpu_total is not None else "N/A"
        results.append(f"{'CPUAlloc / CPUTot':<40}: {output}")

        output = f"{gres}" if gres is not None else "N/A"
        results.append(f"{'Generic Resources':<40}: {output}")

        output = f"{free_mem} / {tot_mem} OR {float(free_mem)/float(tot_mem)*100:.2f}% free" if free_mem is not None and tot_mem is not None else "N/A"
        results.append(f"{'Free Memory / Total Real Memory (GB)':<40}: {output}")

        output = f"{configured_gpus-allocated_gpus}/{configured_gpus} OR {float(configured_gpus-allocated_gpus)/configured_gpus*100:.2f}% free\n" if configured_gpus is not None and allocated_gpus is not None else "N/A"
        results.append(f"{'Free GPUs / Total GPUs':<40}: {output}")

    results = "\n".join(results)

    return results


def calculate_subplot_grid_size(length_of_list):
    sqrt_length = math.sqrt(length_of_list)
    rows = math.floor(sqrt_length)
    columns = rows

    while rows * columns < length_of_list:
        if columns == rows:
            columns += 1
        else:
            rows += 1

    return rows, columns


def PrintSourceCode(function):
    """Returns the source code of a function"""
    src_code = "".join(inspect.getsourcelines(function)[0])
    return Code(src_code, language="python")


def get_function_by_pid(pid):
    """Returns the function or executable associated with a process ID"""
    try:
        process = psutil.Process(pid)
        function = process.name()  # This retrieves the name of the function or executable associated with the process.
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
        with self.assertRaises(AssertionError):
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
    Extracts the numeric part from a filename at a given location.

    Args:
        filename (str): The file name from which to extract the number.
        captured_loc (str): location of capture in filename split by "/"
        prefix/suffix (str): strings immediately before/after the capture

    Returns:
        float: The extracted number, or last number of several
    """

    _filename = filename.split("/")[captured_loc]
    _filename = _filename.replace(prefix, "").replace(suffix, "")

    # creates a list of all individual floats found at the capture point
    match = re.findall(r"[0-9]*[.]?[0-9]+", _filename)

    if len(match) != 0:
        return float(match[-1])
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
    if "[]" not in files and "*" not in files:
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


class Silencer:

    def __init__(self, show_stdout=False, show_stderr=False):
        self.show_stdout = show_stdout
        self.show_stderr = show_stderr

    def __enter__(self):
        if not self.show_stdout:
            # Save the original stdout file descriptor
            self._original_stdout_fd = os.dup(1)
            # Replace stdout with a file descriptor to /dev/null
            self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self._devnull_fd, 1)
        if not self.show_stderr:
            # Save the original stderr file descriptor
            self._original_stderr_fd = os.dup(2)
            # Replace stderr with a file descriptor to /dev/null
            self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self._devnull_fd, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.show_stdout:
            # Restore the original stdout file descriptor
            os.dup2(self._original_stdout_fd, 1)
            os.close(self._devnull_fd)
            os.close(self._original_stdout_fd)
        if not self.show_stderr:
            # Restore the original stderr file descriptor
            os.dup2(self._original_stderr_fd, 2)
            os.close(self._devnull_fd)
            os.close(self._original_stderr_fd)
