import os
import psutil
import subprocess

def get_function_by_pid(pid):
    ''' Returns the function or executable associated with a process ID '''
    try:
        process = psutil.Process(pid)
        function = process.name()  # This retrieves the name of the function or executable associated with the process.
        return function
    except psutil.NoSuchProcess:
        return "Process with PID {} not found.".format(pid)
    except psutil.AccessDenied:
        return "Access denied to process with PID {}.".format(pid)

def get_pid_family():
    '''
    Returns the function or executable associated with the current process ID and its parent process ID

    Example:
        1) python example.py              -> python, bash
        2) mpirun -np 1 python example.py -> python, mpirun
    '''
    pid = os.getpid()
    ppid = os.getppid()
    current = get_function_by_pid(pid)
    parent = get_function_by_pid(ppid)
    return current, parent

def check_nvidia_devices():
    ''' Checks nvidia-smi for devices '''
    try:
        subprocess.check_output(["nvidia-smi"])
        return True, "NVIDIA devices exist on your system."
    except Exception:
        return False, "No NVIDIA devices found."

def remove_all_whitespace(string):
    ''' Removes all whitespace from a string '''
    return string.replace(" ", "")


def prepare_mpigpu(accelerator):
    ''' Sets variables to use MPI and/or GPU if requested'''
    #### MPI ####
    assert(accelerator in ['mpi', 'gpu', 'mpigpu', 'gpumpi', '']), f'Invalid accelerator flag: {accelerator}'
    caller, parent = get_pid_family()
    if ("mpi" in parent or 'mpi' in accelerator) and accelerator!='':
        os.environ['ATI_USE_MPI'] = '1'
        USE_MPI = True
    else:
        os.environ['ATI_USE_MPI'] = '0'
        USE_MPI = False
    print(f'MPI is {"enabled" if USE_MPI else "disabled"}')

    #### GPU ####
    if (check_nvidia_devices()[0] or 'gpu' in accelerator) and accelerator!='' and accelerator!='mpi':
        os.environ['ATI_USE_GPU'] = '1'
        USE_GPU = True
    else:
        os.environ['ATI_USE_GPU'] = '0'
        USE_GPU = False
    print(f'GPU is {"enabled" if USE_GPU else "disabled"}')

    return USE_MPI, USE_GPU
