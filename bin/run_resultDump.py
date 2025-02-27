from pyamptools.utility.IO import loadAllResultsFromYaml
import numpy as np
from rich.console import Console
from pyamptools.utility.general import load_yaml, Timer
import os
import argparse
import pandas as pd
from rich.table import Table

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description="Dump (AmpTools, IFT) results to separate csv files")
    argparser.add_argument("yaml_file", type=str, help="Path to the yaml file containing the results")
    argparser.add_argument("-o", "--output_dir", type=str, help="Path to the output directory")
    argparser.add_argument("-n", "--npool", type=int, help="Number of processes to use by multiprocessing.Pool (only for moments calculation)")
    argparser.add_argument("--skip_moments", action="store_true", help="Skip moments calculation")
    argparser.add_argument("--clean", action="store_true", help="Clean the output directory before running")
    args = argparser.parse_args()
    yaml_file = args.yaml_file
    output_dir = args.output_dir
    npool = args.npool
    skip_moments = args.skip_moments
    clean = args.clean
    
    timer = Timer()
    
    console = Console()
    if output_dir is None:
        yaml_primary = load_yaml(yaml_file)
        base_directory = yaml_primary['base_directory']
        output_dir = base_directory
        console.print(f"\nOutput directory not specified, using YAML file's base directory: {output_dir}\n\n\n")

    # apply_mle_queries = False since we assume user always wants all MLE fit results (where mle_query_1 and mle_query_2 are not applied, see YAML file)
    amptools_df, ift_df, ift_res_df, wave_names, masses, tPrimeBins, bpg, latex_name_dict = loadAllResultsFromYaml(yaml_file, pool_size=npool, skip_moments=skip_moments, clean=clean, apply_mle_queries=False)

    ift_csv_path = os.path.join(base_directory, 'ift_df.csv')
    ift_res_csv_path = os.path.join(base_directory, 'ift_res_results.csv')
    amptools_csv_path = os.path.join(base_directory, 'amptools_results.csv')
    
    print_schema = "\n\n\n## [bold]SUMMARY OF COLLECTED RESULTS[/bold] ##\n"
    if skip_moments:
        print_schema += "## [red]Moments were not calculated as user requested...[/red]\n"
    else:
        print_schema += "## [green]Moments were calculated and dumped to csv also...[/green]\n"
    print_schema += (
        "\n********************************************************************\n"
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
            "    - [bold]'<wave>_<res>' columns[/bold]: Contain intensity values (not just amp^2 due to normalization integrals) for parameteric components.\n"
            "    - [bold]'<wave>_cf' columns[/bold]: Contain intensity values (not just amp^2 due to normalization integrals) for correlated field components.\n"
            "    - [bold]'<wave>' columns[/bold]: Contain intensity values (not just amp^2 due to normalization integrals) for the coherent sum of parameteric and correlated field components.\n"
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
            "    - [bold]'<wave>'[/bold]: Fitted intensity (not amp^2 due to normalization integrals)\n"
            "    - [bold]'<wave> err'[/bold]: Error on fitted intensity values\n"
            "    - [bold]'<wave1> <wave2>'[/bold]: Relative phase between wave1 and wave2\n"
            "    - [bold]'<wave1> <wave2> err'[/bold]: Error on relative phase between wave1 and wave2\n"
            "    - [bold]'Hi_LM'[/bold]: Moment with index i for given LM quantum number. i.e. H0_00 is the $H_{0}(0,0)$ moment.\n"
    )
    
    console.print(print_schema)
    console.print("\n\n********************************************************************")
    console.print(f" Number of t-bins: {len(tPrimeBins)}: {np.array(tPrimeBins)}")
    console.print(f" Number of mass-bins: {len(masses)}: {np.array(masses)}")
    console.print(f" Number of wave names: {len(wave_names)}: {wave_names}")
    
    console.print("\n")
    
    console.print("\n********************************************************************")
    if ift_df is not None:
        console.print(f" [red]Writing IFT DataFrame to csv located at: {ift_csv_path}[/red]")
        console.print(f" Shape of IFT results DataFrame {ift_df.shape} with columns: {ift_df.columns}")
        ift_df.to_csv(ift_csv_path, index=False)
        console.print(f" [red]Writing IFT (Res)onance DataFrame to csv located at: {ift_res_csv_path}[/red]")
        console.print(f" Shape of IFT (Res)onance results DataFrame {ift_res_df.shape} with columns: {ift_res_df.columns}")
        ift_res_df.to_csv(ift_res_csv_path, index=False)
    else:
        console.print(" [red]No IFT results found, skipping...[/red]")
    
    console.print("\n********************************************************************")
    if amptools_df is not None:
        console.print(f" [red]Writing AmpTools DataFrame to csv located at: {amptools_csv_path}[/red]")
        console.print(f" Shape of AmpTools results DataFrame {amptools_df.shape} with columns: {amptools_df.columns}")
        amptools_df.to_csv(amptools_csv_path, index=False)
    else:
        console.print(" [red]No AmpTools results found, skipping...[/red]")
        
    ######################################################################################################################################
    ## We can additionally populate a table of integrated yields_for_table and associated uncertainties (across mass) for every parametric component
    ######################################################################################################################################
    if ift_df is not None:
        console.print("\n\n********************************************************************")
        console.print(" [bold]Sorted Integrated Yields for IFT Parameteric Components -> mean +/- std (rel. unc.)[/bold]")
        parametric_amps = [col for col in ift_df.columns if '_cf' not in col and '_amp' not in col and len(col.split('_'))>2] # ignore correlated field components (_cf) and coherent sum cols
        parametric_intensities = [col.rstrip('_amp') for col in parametric_amps]
        t_bin_centers = ift_df['tprime'].unique()
            
        yields_for_table = [] # dumped to screen
        yields_for_csv = {'t': [], 'component': [], 'mean': [], 'std': []} # dumped to csv
        for it, t_bin_center in enumerate(t_bin_centers):
            ift_df_t = ift_df[np.isclose(ift_df['tprime'], t_bin_center, rtol=1e-5)]
            yields_for_table.append([])
            for i, parametric_intensity in enumerate(parametric_intensities):
                # Grouping by sample, sum over mass, then calculate mean / std produces very different results compared to
                # grouping by mass and summing over samples. This is due to the fact that the samples are entire curves!
                # In my original test case, the relative uncertainties went from 20% to 4% where first number comes from grouping by sample
                sampled_integrated_intensities = ift_df_t.groupby('sample')[parametric_intensity].sum()
                mean = sampled_integrated_intensities.mean()
                std = sampled_integrated_intensities.std()
                yields_for_table[it].append(f"{mean:0.2f} +/- {std:0.2f} ({std/mean:0.2f})")
                yields_for_csv['t'].append(t_bin_center)
                yields_for_csv['component'].append(parametric_intensity)
                yields_for_csv['mean'].append(mean)
                yields_for_csv['std'].append(std)

        pretty_t_bin_centers = [f"t={t_bin_center:0.3f} GeV^2" for t_bin_center in t_bin_centers]
        yields_for_table = pd.DataFrame(yields_for_table, columns=parametric_intensities, index=pretty_t_bin_centers).T
        yields_for_csv = pd.DataFrame(yields_for_csv)
        
        # reorder rows so that the largest yields_for_table (summing mean values across t-bins) are at the top
        fyields = yields_for_table.map(lambda x: float(x.split(' +/- ')[0]))
        yields_for_table = yields_for_table.iloc[np.argsort(fyields.sum(axis=1))[::-1]]

        console_table = Console(record=True)
        table = Table()
        table.add_column(f"Parametric Component", justify="left", style="bold green", no_wrap=True)
        for pretty_t_bin_center in pretty_t_bin_centers:
            table.add_column(f"{pretty_t_bin_center}", justify="left", style="bold green", no_wrap=True)
        for i, parametric_intensity in enumerate(yields_for_table.index):
            table.add_row(parametric_intensity, *yields_for_table.iloc[i])
        console_table.print(table)
        
        # Dump table to csv
        table_dump = os.path.join(base_directory, 'integrated_yields_table.csv')
        console.print(f" [red]Writing table of Integrated yields_for_table of all Parametric Components to csv file at: {table_dump}[/red]")
        yields_for_csv.to_csv(table_dump, index=False, float_format='%0.5f')

        console_table.print("\n\n")

    elapsed_time = timer.read()[2]
    console.print(f" [bold]resultDump| Total time taken: {elapsed_time}[/bold]")
