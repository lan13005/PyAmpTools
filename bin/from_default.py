import os
import argparse
from rich.console import Console
import shutil
from string import Template

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_location", type=str, default=None)
    parser.add_argument("-r", "--reaction", type=str, default='etapi', choices=['etapi'])
    args = parser.parse_args()
    
    console = Console()
    
    output_file = args.output_location
    if output_file is None:
        output_file = os.path.join(os.getcwd(), "main.yaml")
    parent_dir = os.path.dirname(output_file)
    if not os.path.exists(parent_dir) and parent_dir != '':
        os.makedirs(parent_dir)
    
    if os.path.exists(output_file):
        console.print(f"\nFile {output_file} already exists. Exiting.\n", style="bold red")
        exit(1)
    
    if args.reaction == 'etapi':
        default_src = f"{os.environ['PYAMPTOOLS_HOME']}/src/pyamptools/utility/yaml/default_etapi.yaml"
        
        console.print(f"\nDeriving from {default_src}", style="bold blue")
        console.print(f"   updating BASE_DIRECTORY (all results dumped relative to here) to {os.getcwd()}/RESULTS", style="bold blue")
        console.print(f"   writing to {output_file}\n", style="bold blue")
        
        with open(default_src, 'r') as file:
            default_yaml = file.read()
        
        template = Template(default_yaml)
        output_yaml = template.safe_substitute(BASE_DIRECTORY=f"{os.getcwd()}/RESULTS")
        
        with open(output_file, 'w') as file:
            file.write(output_yaml)

    