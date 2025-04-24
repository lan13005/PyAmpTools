import os
import argparse
from rich.console import Console
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_location", type=str, default=os.getcwd())
    parser.add_argument("-r", "--reaction", type=str, default='etapi', choices=['etapi'])
    args = parser.parse_args()
    
    console = Console()
    
    output_file = os.path.join(args.output_location, "main.yaml")
    
    if os.path.exists(output_file):
        console.print(f"File {output_file} already exists. Exiting.", style="bold red")
        exit(1)
    
    if args.reaction == 'etapi':
        default_src = f"{os.environ['PYAMPTOOLS_HOME']}/src/pyamptools/utility/yaml/default_etapi.yaml"
        console.print(f"\nCopying {default_src} to {output_file}\n", style="bold green")
        shutil.copy(default_src, output_file)

    