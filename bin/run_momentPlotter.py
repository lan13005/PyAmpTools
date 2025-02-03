import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import argparse
from pyamptools.utility.general import Timer
from pyamptools.utility.io import loadAllResultsFromYaml
from omegaconf.errors import MissingMandatoryValue
from pyamptools.utility.general import load_yaml, Styler

def plot_moment(moment_name, t, masses, _amptools_df, _ift_df, latex_name_dict, show_samples=True, no_errorbands=False, save_file=None):

    ### SET THE STYLE
    style = Styler()
    style.setPlotStyle()

    fig, ax_intens = plt.subplots()
    
    amptools_df, ift_df = None, None
    if _amptools_df is not None:
        amptools_df = _amptools_df.query(f'tprime == {t}')
    if _ift_df is not None:
        ift_df = _ift_df.query(f'tprime == {t}')
    
    mass_width = np.round(masses[1] - masses[0], 4)
    
    ax_intens.set_ylabel(f"Moment Value / {mass_width} GeV$/c^2$")
    ax_intens.set_xlabel("$m_X$ [GeV$/c^2$]")
    
    ## PLOT THE IFT RESULT
    if ift_df is not None:
        if show_samples:
            for sample in ift_df['sample'].unique():
                df_sample = ift_df.query(f'sample == {sample}')
                moment_value = df_sample[moment_name].values
                moment_value = moment_value * bpg # ift generally uses finer binning than amptools, rescale to match
                label = "IFT Result" if sample == ift_df['sample'].unique()[0] else None
                ax_intens.plot(df_sample['mass'], moment_value, color="xkcd:sea blue", alpha=0.2, zorder=0, label=label)
        if not no_errorbands:
            mean = ift_df.groupby(['tprime', 'mass'])[moment_name].mean().reset_index()
            std  = ift_df.groupby(['tprime', 'mass'])[moment_name].std().reset_index()
            moment_value_low  = (mean[moment_name] - std[moment_name]) * bpg
            moment_value_high = (mean[moment_name] + std[moment_name]) * bpg
            label = '' if show_samples else "IFT Result"
            ax_intens.fill_between(masses, moment_value_low, moment_value_high, color="xkcd:sea blue", alpha=0.2, label=label)
        
    ## PLOT THE AMPTOOLS RESULT
    if amptools_df is not None:
        moment_value = amptools_df[moment_name].values
        ax_intens.scatter(amptools_df['mass'], moment_value, color="black", marker="o", s=20, label='Mass Indep. Result')
    
    ax_intens.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_intens.set_title(latex_name_dict[moment_name])
    
    style.setAxisStyle(ax_intens)
    
    if save_file:
        print(f"momentPlotter| Saving Figure: {save_file}")
        fig.savefig(save_file, bbox_inches="tight")

    plt.close(fig)
    del fig, ax_intens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform maximum likelihood fits (using AmpTools) on all cfg files")
    parser.add_argument("yaml_name", type=str, default="conf/configuration.yaml", help="Path a configuration yaml file")
    parser.add_argument("-n", "--npool", type=int, default=4, help="Number of processes to use by multiprocessing.Pool")
    args = parser.parse_args()
    yaml_name = args.yaml_name
    npool = args.npool

    print("\n---------------------")
    print(f"Running {__file__}")
    print(f"  yaml location: {yaml_name}")
    print("---------------------\n")

    timer = Timer()
    
    yaml_primary = load_yaml(yaml_name)
    yaml_secondary = yaml_primary['nifty']['yaml']
    yaml_secondary = load_yaml(yaml_secondary, resolve=False)

    amptools_df, ift_df, wave_names, masses, tprime_centers, bpg, latex_name_dict = loadAllResultsFromYaml(yaml_name, npool)
    
    if latex_name_dict is None:
        raise ValueError("momentPlotter| latex_name_dict is None. This means that moments were not calculated properly for the ift and amptools dataframes!")
    if amptools_df is not None: 
        print(f"momentPlotter| amptools_df loaded with shape {amptools_df.shape} (generally wider than ift_df since it holds additional columns)")
    else: 
        print("momentPlotter| amptools_df is None. No AmpTools results found.")
    if ift_df is not None: 
        print(f"momentPlotter| ift_df loaded with shape {ift_df.shape}")
    else: 
        print("momentPlotter| ift_df is None. No IFT results found.")

    ####################################################
    #### Plot the moments
    ####################################################
    print("momentPlotter| Plotting moments...")
    # synchronization of yaml pair currently only happens automatically when running `run_ift`
    #   We catch exceptions from OmegaConf for missing values or Python type errors for None type
    try: 
        outputFolder = yaml_secondary['GENERAL']['outputFolder']
        if outputFolder == '???':
            outputFolder = yaml_primary['nifty']['output_directory']
    except (MissingMandatoryValue, TypeError):
        outputFolder = yaml_primary['nifty']['output_directory']
    outputFolder = os.path.join(outputFolder, "plots/moments")
    os.makedirs(outputFolder, exist_ok=True)

    for t in tprime_centers:
        for moment_name in latex_name_dict.keys():
            plot_moment(moment_name, t, masses, amptools_df, ift_df, latex_name_dict, save_file=f"{outputFolder}/{moment_name}_t{t}.pdf")

    print(f"momentPlotter| Elapsed time {timer.read()[2]}\n\n")
