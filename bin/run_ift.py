import argparse
import logging
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import subprocess

from pyamptools.utility.general import Timer, dump_yaml, load_yaml

############################################################################
# This file wraps a call to iftpwa's execution script iftpwa takes an input 
# yaml file
############################################################################

# Keys: are the keys in the Destination yaml file
# Values: Keys in the Source yaml file. SRC prefixed to the key will load info
#         from the source yaml file.
# TODO: For more complicated formattings we might have to make a pythonic system
#       that uses {} to denote variable search paths. ATM we variable interpolate
#       each part separated by /

def get_value_from_src(main_dict, key_path):
    """
    Load value from the source (SRC) yaml file following a period separated key path
    
    Args:
        main_dict (dict): Source dictionary (YAML file)
        key_path (str): Key path in the main_dict, period separated
        
    Returns:
        value: The value in the source yaml file
    """
    if not key_path.startswith("SRC"):
        return key_path
    
    key_path = key_path.split(".")
    n_steps = len(key_path)
    assert key_path[0] == "SRC"
    assert key_path[1] in main_dict, f"Key {key_path[1]} not found in the source yaml file"
    value = main_dict[key_path[1]]
    for i in range(2, n_steps):
        assert key_path[i] in value, f"Key {key_path[i]} not found in the source yaml file"
        value = value[key_path[i]]
    assert "${" not in str(value), f"Value: {value} should not variable substitution"
    return value

def set_nested_value_inplace(nested_dict, keys, value, verbose=True):
    """
    Set value inplace in a nested dictionary for arbitrary number of nestings.
    
    Args:
        nested_dict (dict): The nested dictionary.
        keys (list): List of keys representing the path to the value.
        value: The value to set.
    
    Returns:
        None
    """
    current_level = nested_dict
    for key in keys[:-1]:
        if key not in current_level:
            current_level[key] = {}
        current_level = current_level[key]
    if verbose:
        logger.info(f"* Setting {keys[-1]} to {value}")
    current_level[keys[-1]] = value

if __name__ == "__main__":

    # intercept the help flag to show iftPwaFit help message
    import sys    
    if '-h' in sys.argv or '--help' in sys.argv:
        os.system("iftPwaFit --help")
        exit(0)

    # Optuna had a conflict with CCDB from PYTHONPATH
    _pp = os.environ.get("PYTHONPATH", "")
    if _pp:
        _filtered = []
        for _p in _pp.split(":"):
            if "AMPTOOLS_GENERATORS/ccdb" in _p or "/ccdb/" in _p or _p.endswith("/ccdb"):
                continue
            _filtered.append(_p)
        os.environ["PYTHONPATH"] = ":".join(_filtered)

    parser = argparse.ArgumentParser(description="Perform IFT fit over all cfg files")
    parser.add_argument("main_yaml", type=str, default="conf/configuration.yaml", help="Path to the main yaml file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("additional_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    main_yaml = args.main_yaml
    additional_args = args.additional_args
    
    if '-v' in args.additional_args or '--verbose' in args.additional_args:
        raise ValueError("verbose flag has to be before positional argument! Example pa run_ift -v main.yaml")

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format='%(asctime)s| %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)

    print()
    logger.info("---------------------")
    logger.info(f"Running {__file__}")
    logger.info(f"  yaml location: {main_yaml}")
    logger.info("---------------------\n")

    timer = Timer()
    cwd = os.getcwd()

    if not os.path.exists(main_yaml):
        raise FileNotFoundError(f"YAML file {main_yaml} not found")

    main_dict = load_yaml(main_yaml)
    main_dest = main_dict["NIFTY"]["PWA_MANAGER"]["yaml"]
    
    if main_dict['nifty']['yaml'] is None:
        logger.info("No NIFTy yaml file specified. Exiting...")
        exit(1)
    
    output_directory = os.path.join(main_dict['base_directory'], "NIFTY")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    iftpwa_yaml = main_dict['NIFTY']
    iftpwa_dest = f"{output_directory}/iftpwa.yaml"
    dump_yaml(iftpwa_yaml, iftpwa_dest)
    logger.info(f"Copying {main_yaml} to {main_dest}")
    os.system(f"cp {main_yaml} {main_dest}")

    mpi_processes = main_dict['nifty']['mpi_processes'] if 'mpi_processes' in main_dict['nifty'] else None

    ## NOTE: GPU is not working. Distribution across mpi processes uses too much memory still. 
    ##       Unsure how to better balance. Uncomment the following to try to load GPUs again
    num_gpus = 0
    # if subprocess.call("command -v nvidia-smi", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
    #     get_cmd = "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
    #     num_gpus = int(os.popen(get_cmd).read())
    #     print(f"Number of GPUs found (will attempt to distribute mpi work across all): {num_gpus}")
    prefix = ""
    if mpi_processes is not None and mpi_processes > 1:
        prefix  = f"mpiexec -v -n {mpi_processes} "
        prefix += "--mca btl ^openib " # Disable openib

    add_args = ' '.join(additional_args)
    if add_args != '': 
        add_args = f' {add_args}'
    if num_gpus > 0:
        # Distribute mpi work across all GPUs
        cmd = (f"{prefix}bash -c 'export CUDA_VISIBLE_DEVICES=$(($OMPI_COMM_WORLD_RANK % {num_gpus})); "
            f'export XLA_PYTHON_CLIENT_MEM_FRACTION=1; '
            f'export TF_FORCE_GPU_ALLOW_GROWTH=true; '
            f"iftPwaFit --iftpwa_config {iftpwa_dest}{add_args}'")
    else:
        cmd = (f"{prefix}bash -c 'export JAX_PLATFORMS=cpu; "
                f"iftPwaFit --iftpwa_config {iftpwa_dest}{add_args}'")
    
    print(f"\nmpiexec command:\n {cmd}\n")
    os.system(cmd) # Run IFT pwa fit
    os.system(f"rm -f {iftpwa_dest}") # Cleanup

    logger.info(f"ift| Elapsed time {timer.read()[2]}\n\n")