import argparse
import os
import re

from omegaconf import OmegaConf, dictconfig
from rich.console import Console

from pyamptools.utility.general import get_git_commit_hash_for_package

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull default yaml file into your current directory (with guidance tutorial)")
    parser.add_argument("-o", "--output_file", type=str, default="main.yaml")
    parser.add_argument("-t", "--type", type=str, default='1d', choices=['1d', 'twops', 'vecps'])
    parser.add_argument("-c", "--clean", action='store_true')
    args = parser.parse_args()
    
    console = Console()
    
    output_file = os.path.join(os.getcwd(), args.output_file)
    parent_dir = os.path.dirname(output_file) # This would the cwd so it should always exist
    if not os.path.exists(parent_dir) and parent_dir != '':
        os.makedirs(parent_dir)
    
    if os.path.exists(output_file) and not args.clean:
        console.print(f"\nFile {output_file} already exists. Exiting.\n", style="bold red")
        exit(1)
    
    if args.type == '1d':
        default_src = f"{os.environ['PYAMPTOOLS_HOME']}/src/pyamptools/utility/yaml/default_1d.yaml"
    elif args.type == 'twops':
        default_src = f"{os.environ['PYAMPTOOLS_HOME']}/src/pyamptools/utility/yaml/default_etapi.yaml"
    elif args.type == 'vecps':
        default_src = f"{os.environ['PYAMPTOOLS_HOME']}/src/pyamptools/utility/yaml/default_omegaeta.yaml"
    else:
        console.print(f"\nInvalid CLI argument (-t, --type) passed: {args.type}. Exiting.\n", style="bold red")
        exit(1)
        
    console.print(f"\nDeriving from: {default_src}\nOutputting to: {output_file}", style="bold blue")
    
    with open(default_src, 'r') as file:
        yaml_str = file.read()
        
    # Identify keys user needs to update
    def find_placeholders(config_dict):
        # Must not resolve the variables ${...}
        for key, value in config_dict.items_ex(resolve=False):
            if isinstance(value, str):
                if re.search(r'(?<!\w)(?<!\$\{)_[A-Z0-9]+(?:_[A-Z0-9]+)*_(?!\w)(?!\})', value):
                    console.print(f"{key}: {value}", style="bold yellow")
            if isinstance(value, (dict, dictconfig.DictConfig)):
                find_placeholders(value)
                
    console.print(f"\nPlease update the keys (if any) with placeholders in {output_file}", style="bold red")
    yaml_dict = OmegaConf.load(default_src)
    console.print(f"Placeholder keys in yaml file:", style="bold underline red")
    find_placeholders(yaml_dict)
    
    console.print(f"\nYou will also need to update top-level keys to match your reaction and kinematics", style="bold red")
    console.print(f"For instance:", style='bold underline red')
    console.print(f"waveset", style='bold yellow')
    console.print(f"min_mass", style='bold yellow')
    console.print(f"max_mass", style='bold yellow')
    console.print(f"n_mass_bins", style='bold yellow')
    console.print(f"min_t", style='bold yellow')
    console.print(f"max_t", style='bold yellow')
    console.print(f"n_t_bins", style='bold yellow')
    console.print(f"daughters", style='bold yellow')
    
    # Append the latest commit hashes to the yaml file
    pyamptools_commit_hash = get_git_commit_hash_for_package("pyamptools")
    iftpwa_commit_hash = get_git_commit_hash_for_package("iftpwa1")
    yaml_str += f"pyamptools_commit_hash: {pyamptools_commit_hash}\n"
    yaml_str += f"iftpwa_commit_hash: {iftpwa_commit_hash}\n"
    
    with open(output_file, 'w') as file:
        file.write(yaml_str)

    console.print(f"\nDone!", style="bold green")