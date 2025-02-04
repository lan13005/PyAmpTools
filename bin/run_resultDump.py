from pyamptools.utility.io import loadAllResultsFromYaml
import numpy as np
from rich.console import Console
from pyamptools.utility.general import load_yaml
import os
import argparse

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description="Dump (AmpTools, IFT) results to separate csv files")
    argparser.add_argument("yaml_file", type=str, help="Path to the yaml file containing the results")
    argparser.add_argument("-o", "--output_dir", type=str, help="Path to the output directory")
    args = argparser.parse_args()
    yaml_file = args.yaml_file
    output_dir = args.output_dir
    
    console = Console()
    
    if output_dir is None:
        yaml_primary = load_yaml(yaml_file)
        base_directory = yaml_primary['base_directory']
        output_dir = base_directory
        console.print(f"\nOutput directory not specified, using YAML file's base directory: {output_dir}\n\n\n")

    amptools_df, ift_df, ift_res_df, wave_names, masses, tPrimeBins, bpg, latex_name_dict = loadAllResultsFromYaml(yaml_file)

    ift_csv_path = os.path.join(base_directory, 'ift_results.csv')
    ift_res_csv_path = os.path.join(base_directory, 'ift_res_results.csv')
    amptools_csv_path = os.path.join(base_directory, 'amptools_results.csv')
    
    print_schema = (
        "\n\n\n## [bold]SUMMARY OF COLLECTED RESULTS[/bold] ##\n\n"
        "[bold]SCHEMA:[/bold]\n"
        "- [cyan]t-bins / mass-bins:[/cyan] Bin centers.\n"
    )
    if ift_df is not None:
        print_schema += (
            "- [cyan]IFT DataFrame:[/cyan] Quantities per (t, mass, posterior sample).\n"
            "    - [bold]'intensity_corr'[/bold]: Fitted acceptance corrected intensity values. Aggregate over samples to get mean/std.\n"
            "    - [bold]'intensity'[/bold]: Fitted intensity values. Aggregate over samples to get mean and standard deviation.\n"
            "    - [bold]'<wave>_<res>_amp' columns[/bold]: Contain complex amplitude values for parameteric components.\n"
            "    - [bold]'<wave>_cf_amp' columns[/bold]: Contain complex amplitude values for correlated field components.\n"
            "    - [bold]'<wave>_amp' columns[/bold]: Contain complex amplitude values for the coherent sum of parameteric and correlated field components.\n"
            "    - [bold]'<wave>_<res>' columns[/bold]: Contain intensity values for parameteric components.\n"
            "    - [bold]'<wave>_cf' columns[/bold]: Contain intensity values for correlated field components.\n"
            "    - [bold]'<wave>' columns[/bold]: Contain intensity values for the coherent sum of parameteric and correlated field components.\n"
            "    - [bold]'Hi_LM'[/bold]: Moment with index i for given LM quantum number. i.e. H0_00 is the $H_{0}(0,0)$ moment.\n"
            "- [cyan]IFT (Res)onance DataFrame:[/cyan] Quantities per (sample).\n"
            "    - [bold]'index' of the DataFrame[/bold]: Sample index\n"
            "    - [bold]'<prior_parameter_name>' columns[/bold]: Contain the value of the resonance parameter.\n"
            ""
        )
    if amptools_df is not None:
        print_schema += (
            "- [cyan]AmpTools DataFrame:[/cyan] Quantities per (t, mass, random initialization).\n"
            "    - [bold]'nll'[/bold]: AmpTools.FitResults.likelihood() return value\n"
            "    - [bold]'iteration'[/bold]: random initialization iteration number\n"
            "    - [bold]'status'[/bold]: minuit return status (0=success)\n"
            "    - [bold]'ematrix'[/bold]: error matrix status (3=success)\n"
            "    - [bold]'<wave>_amp'[/bold]: Fitted complex amplitude values\n"
            "    - [bold]'<wave>'[/bold]: Fitted intensity values\n"
            "    - [bold]'<wave> err'[/bold]: Error on fitted intensity values\n"
            "    - [bold]'<wave1> <wave2>'[/bold]: Relative phase between wave1 and wave2\n"
            "    - [bold]'<wave1> <wave2> err'[/bold]: Error on relative phase between wave1 and wave2\n"
            "    - [bold]'Hi_LM'[/bold]: Moment with index i for given LM quantum number. i.e. H0_00 is the $H_{0}(0,0)$ moment.\n"
    )
    
    console.print(print_schema)
    console.print(f" Number of t-bins: {len(tPrimeBins)}: {np.array(tPrimeBins)}")
    console.print(f" Number of mass-bins: {len(masses)}: {np.array(masses)}")
    console.print(f" Number of wave names: {len(wave_names)}: {wave_names}")
    
    console.print("\n")
    
    if ift_df is not None:
        console.print(f" [red]Writing IFT DataFrame to csv located at: {ift_csv_path}[/red]")
        console.print(f" Shape of IFT results DataFrame {ift_df.shape} with columns: {ift_df.columns}")
        ift_df.to_csv(ift_csv_path, index=False)
        console.print(f" [red]Writing IFT (Res)onance DataFrame to csv located at: {ift_res_csv_path}[/red]")
        console.print(f" Shape of IFT results DataFrame {ift_res_df.shape} with columns: {ift_res_df.columns}")
        ift_res_df.to_csv(ift_res_csv_path, index=False)
    else:
        console.print(" [red]No IFT results found, skipping...[/red]")
    
    if amptools_df is not None:
        console.print(f" [red]Writing AmpTools DataFrame to csv located at: {amptools_csv_path}[/red]")
        console.print(f" Shape of AmpTools results DataFrame {amptools_df.shape} with columns: {amptools_df.columns}")
        amptools_df.to_csv(amptools_csv_path, index=False)
    else:
        console.print(" [red]No AmpTools results found, skipping...[/red]")
