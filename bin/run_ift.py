import argparse
import logging
import os
import subprocess

from omegaconf import OmegaConf

from pyamptools.utility.general import Timer, dump_yaml, load_yaml

############################################################################
# This file wraps a call to iftpwa's execution script iftpwa takes an input 
# yaml file. pyamptools also orchestrates several operations into a yaml file
# This script synchronizes the two yaml files by loading the iftpwa yaml file
# and updating it with values from the pyamptools yaml file.
# The whole chain will then operate under the same operating conditions
############################################################################

# Keys: are the keys in the Destination yaml file
# Values: Keys in the Source yaml file. SRC prefixed to the key will load info
#         from the source yaml file.
# TODO: For more complicated formattings we might have to make a pythonic system
#       that uses {} to denote variable search paths. ATM we variable interpolate
#       each part separated by /
mappings = {
    "GENERAL.outputFolder": "SRC.nifty.output_directory",
}

def get_value_from_src(src_yaml, key_path):
    """
    Load value from the source (SRC) yaml file following a period separated key path
    
    Args:
        src_yaml (dict): Source dictionary (YAML file)
        key_path (str): Key path in the src_yaml, period separated
        
    Returns:
        value: The value in the source yaml file
    """
    if not key_path.startswith("SRC"):
        return key_path
    
    key_path = key_path.split(".")
    n_steps = len(key_path)
    assert key_path[0] == "SRC"
    assert key_path[1] in src_yaml, f"Key {key_path[1]} not found in the source yaml file"
    value = src_yaml[key_path[1]]
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

    parser = argparse.ArgumentParser(description="Perform IFT fit over all cfg files")
    parser.add_argument("yaml_name", type=str, default="conf/configuration.yaml", help="Path a configuration yaml file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("additional_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    yaml_name = args.yaml_name
    additional_args = args.additional_args
    
    if '-v' in args.additional_args or '--verbose' in args.additional_args:
        raise ValueError("verbose flag has to be before positional argument! Example pa run_ift -v pyamptools.yaml")

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format='%(asctime)s| %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)

    print()
    logger.info("---------------------")
    logger.info(f"Running {__file__}")
    logger.info(f"  yaml location: {yaml_name}")
    logger.info("---------------------\n")

    timer = Timer()
    cwd = os.getcwd()

    if not os.path.exists(yaml_name):
        raise FileNotFoundError(f"YAML file {yaml_name} not found")

    src_yaml = load_yaml(yaml_name)
    
    if src_yaml['nifty']['yaml'] is None:
        logger.info("No NIFTy yaml file specified. Exiting...")
        
    dest_name = src_yaml['nifty']['yaml']
    synchronize = src_yaml['nifty']['synchronize']
    output_directory = src_yaml['nifty']['output_directory']
    dest_yaml = load_yaml(dest_name, resolve=False)

    if synchronize:
        print()
        logger.info("------------------ SYNCHRONIZING YAML FILES ------------------")
        logger.info(f"Base YAML file used for IFTPWA: {dest_name}")
        logger.info(f"    will be synchronized with values found in your provided yaml file: {yaml_name}")
        for dest_keys, src_keys in mappings.items():
            dest_keys = dest_keys.split(".")
            src_keys = src_keys.split("/")
            src_keys = [get_value_from_src(src_yaml, key_path) for key_path in src_keys]
            if len(src_keys) == 1:
                src_keys = src_keys[0]
            else:
                src_keys = "/".join(src_keys)
            set_nested_value_inplace(dest_yaml, dest_keys, src_keys, verbose=True)
        logger.info("--------------------------------------------------------------\n")
    else:
        logger.info(f"Base YAML file used for IFTPWA: {dest_name}")

    dump_yaml(dest_yaml, ".nifty.yaml") # Create "synchronized" yaml file
    if output_directory is not None and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    logger.info(f"Copying {yaml_name} to {output_directory}/.{os.path.basename(yaml_name)}")
    os.system(f"cp {yaml_name} {output_directory}/.{os.path.basename(yaml_name)}") # Copy the source yaml file to the output directory

    mpi_processes = src_yaml['nifty']['mpi_processes'] if 'mpi_processes' in src_yaml['nifty'] else None

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
            f"iftPwaFit --iftpwa_config .nifty.yaml{add_args}'")
    else:
        cmd = (f"{prefix}bash -c 'export JAX_PLATFORMS=cpu; "
                f"iftPwaFit --iftpwa_config .nifty.yaml{add_args}'")
    
    print(f"\nmpiexec command:\n {cmd}\n")
    os.system(cmd) # Run IFT pwa fit

    # TODO: Need to support prior simulation
    # Plotting should also be done
    # os.system(f"iftPwaFit --iftpwa_config .nifty.yaml --prior_simulation")
    
    os.system("rm -f .nifty.yaml") # Cleanup

    logger.info(f"ift| Elapsed time {timer.read()[2]}\n\n")