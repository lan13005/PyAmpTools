from pyamptools import atiSetup
from pyamptools.utility.general import prettyLabels  # Maps i.e. Sp0+ to $S_{0}^{+}$
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import argparse

##############################################################
# This (DX) diagnostic script is used to plot the normalization
# integrals and amplitude integrals as a heatmap and if there
# are multiple bins (suggested by the format string) then it
# will all of them and track the mass dependence of all matrix elements
#
# NOTE: CHECK CALCULATIONS IN THE FUTURE
# Amptools splits Re/Im parts of the amps into separate sectors
# We need to check whether the calculation is correct. Currently
# we take the Integrals for real part add them to the integrals
# for the imaginary part (to form a complex number) and then take
# the absolute value. The integrals from the 4 polarizations are
# summed together. We then normalize the diagonal by dividing by
# sqrt(I_ii) * sqrt(I_jj). So the resulting matrix should be
# an average integral matrix over the 4 polarizations.
##############################################################


def load_integrals(fitFile, i_mass=None, symmetry_mask=True, plot_heatmap=False):
    """
    Args:
        fitFile: str
            Path to the fit file containing the results
        i: int
            Index of the mass bin for output file naming. If None, the output file will be named 'integrals.pdf'
        symmetry_mask: bool
            Mask diagonal and upper triangle due to normalization and symmetry
        plot_heatmap: bool
            Whether to plot the integrals as heatmaps
    """

    ###############################################################################################
    # LOAD FIT RESULTS OBJECT AND EXTRACT AMPLITUDE NAMES
    ###############################################################################################

    if not os.path.isfile(fitFile):
        print(f"{fitFile} does not exist. Exiting...")
        return

    results = FitResults(fitFile)
    if not results.valid():
        print(f"Invalid fit result in file: {fitFile}")
        return

    #### OPTION 1: ####
    # Filter production amplitudes map to select only unique ones (one representative)
    # unique_amps = {}
    # for amp, val in dict(results.ampProdParMap()).items():
    #     if val not in unique_amps:
    #         unique_amps[val] = amp
    # ampList = list(unique_amps.values())
    #### OPTION 2: ####
    ampList = [str(x) for x in results.ampList()]

    # Create map: reactions -> amplitudes. Amplitudes are full names Reaction::Sum::Amp
    reaction_amp_map = {}
    for amp in ampList:
        reaction = amp.split("::")[0]
        if reaction not in reaction_amp_map:
            reaction_amp_map[reaction] = []
        reaction_amp_map[reaction].append(amp)
    reaction_amp_map = {k: sorted(v, key=lambda x: x[-1]) for k, v in reaction_amp_map.items()}

    ###############################################################################################
    # LOAD INTEGRALS SUM OVER RE/IM AND REACTIONS
    ###############################################################################################

    normInts = {}
    ampInts = {}
    for reaction, amps in reaction_amp_map.items():
        # Hypothesis
        # Perhaps why we see only real integral matricies is due to the amptools implementation
        #   where we separate Real and imaginary parts
        normInt = np.empty((len(amps), len(amps)))
        ampInt = np.empty((len(amps), len(amps)))

        for i, amp1 in enumerate(amps):
            for j, amp2 in enumerate(amps):
                # With acceptance effects
                integral = results.normInt(reaction).normInt(amp1, amp2)
                integral = integral if integral.imag != 0 else integral.real
                normInt[i][j] = integral

                # Perfect acceptance
                integral = results.normInt(reaction).ampInt(amp1, amp2)
                integral = integral if integral.imag != 0 else integral.real
                ampInt[i][j] = integral

        # Merge integrals from Re and Im parts
        re_mask = ["Re" in amp for amp in amps]
        re_mask = np.outer(re_mask, re_mask)
        im_mask = ["Im" in amp for amp in amps]
        im_mask = np.outer(im_mask, im_mask)
        re_amps = [amp for amp in amps if "Re" in amp]
        dim = len(re_amps)
        normInt = normInt[re_mask].reshape(dim, dim) + 1j * normInt[im_mask].reshape(dim, dim)
        ampInt = ampInt[re_mask].reshape(dim, dim) + 1j * ampInt[im_mask].reshape(dim, dim)
        normInt = np.abs(normInt)
        ampInt = np.abs(ampInt)

        normInts[reaction] = normInt
        ampInts[reaction] = ampInt

    # Create a total summed integral matrix that sums over reactions
    normInts["total"] = sum(normInts.values())
    ampInts["total"] = sum(ampInts.values())

    # All integrals have the same shape therefore same masking
    mask = np.triu(np.ones_like(normInts["total"], dtype=bool))

    for integralMatrices in [normInts, ampInts]:
        for reaction, integralMatrix in integralMatrices.items():
            # Normalize the diagonal dividing row-wise by sqrt(I_ii) and column-wise by sqrt(I_jj)
            diag_sqrts = np.sqrt(np.diag(integralMatrix))
            integralMatrix = integralMatrix / diag_sqrts[:, None] / diag_sqrts[None, :]

            if symmetry_mask:
                integralMatrix[mask] = 0

            integralMatrices[reaction] = integralMatrix

    ###############################################################################################
    # PLOT HEATMAPS
    ###############################################################################################

    ampNames = list(reaction_amp_map.values())[0]
    ampNames = [amp for amp in ampNames if "Re" in amp]
    ampNames = [amp.split("::")[-1] for amp in ampNames]  # ~ Amp

    if plot_heatmap:
        for reaction in ["total"]:  # normInts.keys():
            if not reaction == "total":
                ampNames = reaction_amp_map[reaction]
                ampNames = [amp for amp in ampNames if "Re" in amp]
                ampNames = ["::".join(np.array(amp.split("::"))[[0, 2]]) for amp in ampNames]  # ~ Reaction::Amp

            matrix_size = len(ampNames)

            max_val = max(np.max(np.abs(normInts[reaction])), np.max(np.abs(ampInts[reaction])))

            fig_size = max(10, matrix_size * 0.5)
            fig_size = (fig_size, fig_size)
            font_size = min(16, 240 // matrix_size)  # Cap font size at 16 and scale inversely with matrix size
            annot_font_size = min(12, 120 // matrix_size)  # Cap font size at 12 and scale inversely with matrix size

            shared_kwargs = {
                "mask": mask,
                "cmap": "coolwarm",
                "xticklabels": ampNames,
                "yticklabels": ampNames,
                "annot": True,
                "annot_kws": {"fontsize": annot_font_size},  # Dynamic font size for annotations
                "vmin": 0,
                "vmax": max_val,
            }

            ofile_tag = f"_{i_mass}" if i_mass is not None else ""

            fig, axes = plt.subplots(1, 1, figsize=fig_size)
            axes.set_box_aspect(1)
            sns.heatmap(normInts[reaction], ax=axes, **shared_kwargs)
            plt.setp(axes.get_xticklabels(), fontsize=font_size)
            plt.setp(axes.get_yticklabels(), fontsize=font_size)
            axes.set_title("Normalization Integral", size=20)
            plt.savefig(f"{output_folder}/normint{ofile_tag}.pdf")

            fig, axes = plt.subplots(1, 1, figsize=fig_size)
            sns.heatmap(ampInts[reaction], ax=axes, cbar=False, **shared_kwargs)
            axes.set_box_aspect(1)
            plt.setp(axes.get_xticklabels(), fontsize=font_size)
            plt.setp(axes.get_yticklabels(), fontsize=font_size)
            axes.set_title("Amplitude Integral", size=20)
            plt.savefig(f"{output_folder}/ampint{ofile_tag}.pdf")

            plt.close("all")

    return normInts, ampInts, ampNames


def plot_integral_mass_dependence(min_mass, max_mass, n_mass_bins, fitFileFmt, symmetry_mask=True, plot_heatmap=False):
    """
    Args:
        min_mass: float
            Minimum mass
        max_mass: float
            Maximum mass
        n_mass_bins: int
            Number of mass bins
        fitFileFmt: str
            String format with a SINGLE format specifier for the mass bin (i)ndex. For example /result/folder_{i}/bin_{i}.fit
        symmetry_mask: bool
            Mask diagonal and upper triangle due to normalization and symmetry
        plot_heatmap: bool
            Whether to plot the integrals as heatmaps
    """

    if n_mass_bins is None:
        raise ValueError("Need to know how many bins you have. Please pass it as an argument to '--n_mass_bins'. Exiting...")

    if min_mass is None or max_mass is None:
        print("You did not provide the mass binning information. Output plots will assume x-axis is the mass bin index\n")
        masses = np.arange(n_mass_bins)
    else:
        masses = np.linspace(min_mass, max_mass, n_mass_bins + 1)
        masses = 0.5 * (masses[1:] + masses[:-1])

    normInt_pair_values = {}
    ampInt_pair_values = {}

    for i in range(n_mass_bins):
        normInts, ampInts, ampNames = load_integrals(fitFileFmt.format(i=i), i_mass=i, symmetry_mask=symmetry_mask, plot_heatmap=plot_heatmap)

        N = len(ampNames)

        # We kept lower triangle
        for j in range(N):
            for i in range(j + 1, N):
                amp1 = prettyLabels[ampNames[i]] if ampNames[i] in prettyLabels else ampNames[i]
                amp2 = prettyLabels[ampNames[j]] if ampNames[j] in prettyLabels else ampNames[j]
                pair_name = f"[{amp1}, {amp2}]"
                normInt_pair_value = normInts["total"][i][j]
                ampInt_pair_value = ampInts["total"][i][j]
                if pair_name not in normInt_pair_values:
                    normInt_pair_values[pair_name] = []
                    ampInt_pair_values[pair_name] = []
                normInt_pair_values[pair_name].append(normInt_pair_value)
                ampInt_pair_values[pair_name].append(ampInt_pair_value)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    for i, integral_pair_values, label in zip(range(2), [normInt_pair_values, ampInt_pair_values], ["Normalization", "Amplitude"]):
        for pair_name, pair_values in integral_pair_values.items():
            if sum(pair_values) > 0:
                axes[i].plot(masses, pair_values, label=pair_name)

        axes[i].tick_params(axis="both", labelsize=16)  # Set the size of y-axis ticks
        axes[i].set_xlabel("Mass (GeV)", size=24)
        axes[i].set_ylabel(f"{label} Integral Matrix Element", size=24)
        axes[i].set_ylim(0)

    plt.tight_layout()
    axes[1].legend()
    plt.savefig(f"{output_folder}/matrix_element_mass_dependence.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make diagnostic plots for (norm)alization and (amp)litude integrals and track mass dependence")
    parser.add_argument("fit_file", type=str, help="Path to the fit file containing the results. Accepts a format string with a single format specifier for the mass bin (i)ndex. For example /result/folder_{i}/bin_{i}.fit")
    parser.add_argument("--n_mass_bins", default=None, type=int, help="Number of mass bins. Will loop over mass (i)ndices.")
    parser.add_argument("--min_mass", default=None, type=float, help="Minimum mass. Only used for x-axis domain.")
    parser.add_argument("--max_mass", default=None, type=float, help="Maximum mass. Only used for x-axis domain.")
    parser.add_argument("--output_folder", type=str, default="./dx_integrals", help="Folder to save the plots")
    parser.add_argument("--no_heatmap", action="store_true", default=False, help="Turn off plotting of the integrals as heatmaps, i.e. if you only want to track mass dependence of the elements")
    args = parser.parse_args()

    atiSetup.setup(globals())

    fit_file = args.fit_file
    min_mass = args.min_mass
    max_mass = args.max_mass
    n_mass_bins = args.n_mass_bins
    plot_heatmap = not args.no_heatmap
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if plot_heatmap:
        print(f"\nHeatmaps will be saved to {output_folder}")
    else:
        print("\nHeatmaps will not be plotted")

    # If user input fit_file looks a string format with a single format specifier for the mass bin (i)ndex we will loop over all bins
    #   else assume its a single fit file
    if "{i}" in fit_file:
        print("fit_file format looks like multiple bins were requested to be diagnosed...\n")
        plot_integral_mass_dependence(min_mass, max_mass, n_mass_bins, fit_file, plot_heatmap=plot_heatmap)
    else:
        print("fit_file format looks like a single bin was requested to be diagnosed...\n")
        load_integrals(fit_file, plot_heatmap=plot_heatmap)
