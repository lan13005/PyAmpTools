import os
import sys
import glob
import multiprocessing
import pickle as pkl
import matplotlib.pyplot as plt

import numdifftools as nd
import numpy as np
import uproot

from pyamptools import atiSetup
from pyamptools.utility.load_parameters import LoadParameters
from pyamptools.utility.general import calculate_subplot_grid_size, prettyLabels
from pyamptools.utility.ift.plotting import KDE


#######################################################
### This utility file contains objects that require ###
### AmpTools / ROOT to be setup in the environment. ###
#######################################################


class DiagnosticCallback:
    """
    This callback will be called after each iteration of the minimization.
    """

    def __init__(
        self,
        operator,
        ati_df,
        masses,
        output_directory,
        resume,
        phase_reference=None,
        daemonManager=None,
        skip_nifty_plot=False,
        draw_bands_else_samples=False,
        intensity_logy=False,
        gen_root_file="",
        gen_hist_fmt="",
        gen_scale_factor=1.0,
        kde_kwargs={},
        mle_kwargs={},
    ):
        self.operator = operator
        self.ati_df = ati_df
        self.masses = masses
        self.output_directory = output_directory
        self.resume = resume
        self.phase_reference = phase_reference
        self.daemonManager = daemonManager
        self.skip_nifty_plot = skip_nifty_plot
        self.draw_bands_else_samples = draw_bands_else_samples
        self.intensity_logy = intensity_logy
        self.gen_root_file = gen_root_file
        self.gen_hist_fmt = gen_hist_fmt
        self.gen_scale_factor = gen_scale_factor
        self.kde_kwargs = kde_kwargs
        self.mle_kwargs = mle_kwargs

        """
        Args:
            operator (Operator): NIFTy operator (i.e. correlated field)
            ati_df (pandas.DataFrame): DataFrame of amptools binned fit results
            masses (List): list of masses used in the fit
            output_directory (str): path to output directory (will be created if it doesn't exist
            resume (bool): whether to overwrite output directory if it already exists
            phase_reference (str): Underscore separated phase reference waves to use for each reflectivity
            daemonManager (DaemonManager): DaemonManager instance that gathers intensities and phase differences
            skip_nifty_plot (bool): whether to skip plotting NIFTy extracted intensities and phase differences
            draw_bands_else_samples (bool): whether to draw the posterior samples directly as lines or bands
            gen_root_file (str): path to root file containing generated distributions
            gen_hist_fmt (str): string format for histogram names in gen_root_file to plug in amplitude names
            gen_scale_factor (float): scale factor to apply to generated distributions
            kde_kwargs (dict): kwargs to pass to the KDE class's plot function
            mle_kwargs (dict): kwargs to pass to to the plotting function of the MLE fit results
        """

        if os.path.exists(self.output_directory) and not self.resume:
            print("DiagnosticCallback| Output directory already exists! Overwriting...")
            os.system(f"rm -rf {self.output_directory}")
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self.fname_fmt = None

        ########################################
        ####### Get list of amplitude names ####
        ########################################

        # Only first bin, since all bins same amps
        fullAmpNames = daemonManager.gather_quantity("amplitudes", None)[0] if daemonManager is not None else []

        # Recall each amplitude belongs to a Reaction and Sum. Full amplitude name -> Reaction::Sum::Amplitude
        # Group amplitudes together that share the same Amplitude name. This logic was designed to sum the
        # contributions from different polarized datasets.
        # Note: this will be problematic for incoherent backgrounds that live in different Sums unless
        #       we allow for non-standard Amplitude names.
        self.amplitudesMap = {}
        for fullAmpName in fullAmpNames:
            ampName = fullAmpName.split("::")[-1]
            if ampName not in self.amplitudesMap:
                self.amplitudesMap[ampName] = [fullAmpName]
            else:
                self.amplitudesMap[ampName].append(fullAmpName)

        if self.gen_root_file != "":
            self._gen = uproot.open(self.gen_root_file)

        unique_refls_found = set([amp[-1] for amp in fullAmpNames])  # Reflectivity sign is always last
        assert len(unique_refls_found) <= 2, "Unexpected number of reflectivities found: {}".format(unique_refls_found)  # amplitude name error?
        if phase_reference != "":
            phase_reference = phase_reference.split("_")
            assert len(phase_reference) == len(unique_refls_found), "You must pass a phase_reference for each reflectivity found in your amptools config file"
            unique_refls_found = sorted(unique_refls_found)  # sort by refl
            phase_reference = sorted(phase_reference, key=lambda x: x[-1])  # sort by refl
            self.phase_reference = {ref: amp for ref, amp in zip(unique_refls_found, phase_reference)}

        print(f"DiagnosticCallback| Phase reference map: {self.phase_reference}")

        self.pkl_results = None  # will be filled with intensities / phases from NIFTy that will be dumped to a pkl file

        self.ext = "pdf"  # might crash if not 'png' and creating GIFs

        print("DiagnosticCallback| Initialized")

    # optimize_kl takes inspect_callback as an argument. inspect_callback will be called after every global iteration.
    def __call__(
        self,
        samples,
        global_iteration,
    ):
        """
        Draws a diagnostic plot of the intensities and phases

        Args:
            samples: SampleList object
            global_iteration: int (NIFTy will pass an int but we will accept a str for more flexible usage)
        """

        if isinstance(global_iteration, int):
            self.fname_fmt = "{}/{}_step{:03d}." + self.ext  # i.e. OutputDir/Intensity_step001.png, OutputDir/Phase_step123.png, etc.
        elif isinstance(global_iteration, str):
            self.fname_fmt = "{}/{}_{}." + self.ext
        else:
            raise ValueError("global_iteration must be an int or str!")

        masses = self.masses
        ati_df = self.ati_df

        ########################
        # Draw MLE fitted intensities and Generated (if available)
        ########################

        mle_alpha = 1.0
        mle_zorder = 6  # MLE fitted scattered intensities (higher zorder = draw on top)
        gen_zorder = 5  # generated distribution
        ift_zorder = 4  # fitted distribution from NIFTy

        _mle_kwargs = {"fmt": "o", "c": "black", "alpha": mle_alpha, "zorder": mle_zorder, "markersize": 6}
        _mle_kwargs.update(self.mle_kwargs)

        meta_cols = ["nll", "delta_nll", "mass", "iteration", "status", "ematrix"]
        extra_ignores = [" ", "err", "total"]

        partial_amp_names = [amp for amp in set(ati_df.columns) - set(meta_cols) if all([key not in amp for key in extra_ignores])]
        # Sort by reflectivity, sorted will make '+' come first
        partial_amp_names = sorted(partial_amp_names, key=lambda x: x[-1])

        print(f"Partial Amplitudes: {partial_amp_names}")

        # Create enough subplots that is close enough to a square given the length of partial_amp_names

        nrows, ncols = calculate_subplot_grid_size(len(partial_amp_names))

        intensity_fig, intensity_axes = plt.subplots(nrows, ncols, figsize=(18, 15), sharey=True, sharex=True)
        intensity_fig.subplots_adjust(wspace=0, hspace=0)
        intensity_axes = intensity_axes.flatten()

        if self.phase_reference != "":
            phase_fig, phase_axes = plt.subplots(nrows, ncols, figsize=(18, 15), sharey=True, sharex=True)
            phase_fig.subplots_adjust(hspace=0, wspace=0)
            phase_axes = phase_axes.flatten()

        if self.phase_reference != "":
            non_reference_waves = [amp for amp in partial_amp_names if amp not in self.phase_reference.values()]
            non_reference_waves = sorted(non_reference_waves, key=lambda x: x[-1])  # sort by reflectivity

        for iax, amp in enumerate(partial_amp_names):
            ax = intensity_axes[iax]

            # Draw Total Intensity First
            binWidth = masses[1] - masses[0]
            counts = ati_df.groupby("mass").first()["total"].values
            edges = np.linspace(masses[0] - binWidth / 2, masses[-1] + binWidth / 2, len(masses) + 1)

            ax.stairs(counts, edges, fill=False, color="black", alpha=1)
            ax.set_xlim(masses[0], masses[-1])
            ax.set_ylim(0, counts.max() * 1.2)  # max all bins

            if self.intensity_logy:
                ax.set_yscale("log")

            # Draw MLE fitted intensities
            ax.errorbar(ati_df["mass"], ati_df[amp], yerr=ati_df[f"{amp} err"], **_mle_kwargs)
            print(f"amp {amp} with label {prettyLabels[amp]}")
            color = "blue" if amp[-1] == "+" else "red"
            ax.text(0.1, 0.85, prettyLabels[amp], transform=ax.transAxes, fontsize=32, c=color, zorder=9)

            # Draw Generated intensities
            if self.gen_root_file != "" and self.gen_hist_fmt != "":
                hist_name = self.gen_hist_fmt.format(amp)
                gen_centers = self._gen[hist_name].axis().centers()
                gen_counts = self._gen[hist_name].values() * self.gen_scale_factor
                ax.plot(gen_centers, gen_counts, color="tab:blue", zorder=gen_zorder, linewidth=3)

        ########################
        # Draw MLE phases
        ########################

        if self.phase_reference != "":
            pairs = []
            for iax, amp in enumerate(non_reference_waves):
                refl = amp[-1]
                reference = self.phase_reference[refl]
                ax = phase_axes[iax]
                pair = f"{reference} {amp}"

                if pair not in ati_df.columns:
                    raise ValueError(f"Pair: {pair} not found in ati_df.columns: {list(ati_df.columns)}")

                mle_phases = ati_df[pair]
                # mle_phases -= circmean( mle_phases, low = -2*np.pi, high = 2*np.pi ) # shift by the circular mean
                mle_phases = np.rad2deg(mle_phases)
                mle_phase_errs = np.rad2deg(ati_df[f"{pair} err"])
                ax.errorbar(ati_df["mass"], mle_phases, yerr=mle_phase_errs, **_mle_kwargs)
                ax.axhline(180, c="black", linestyle="--", linewidth=1, alpha=0.5)
                ax.axhline(-180, c="black", linestyle="--", linewidth=1, alpha=0.5)
                ax.set_xlim(masses[0], masses[-1])
                pairs.append(pair)

        #################################################################################
        ## Loop over samples and calculate intensities and phase differences from NIFTy
        ##   Then plot the mean and std of the samples
        #################################################################################

        if not self.skip_nifty_plot:
            nifty_intensities = []  # List of Dicts of kinematically binned intensities. List over posterior samples
            nifty_amplitudes = []
            nifty_phases = []

            sampleValues = [s.val for s in samples.iterator(self.operator)]
            for samples in sampleValues:
                # Load back in the state by using ati.likelihood() which should recalculate everything
                #   include normalization integrals
                samples = samples.reshape(-1, order="F")
                self.daemonManager.gather_likelihood(0, samples)

                ## CALCULATE THE INTENSITIES
                # ASSUMES: currently all config files have the same amplitudes!
                _intensities = {}
                for ampName, amplitudes in self.amplitudesMap.items():
                    _intensities[ampName] = self.daemonManager.gather_quantity("intensity", amplitudes)
                nifty_intensities.append(_intensities)

                ## GATHER COMPLEX AMPLITUDE VALUES (this might be problematic for incoherent backgrounds also)
                # Assume the elements of amplitudesMap share the complex value (this is the case for the
                # constraining of different polarized datasets and some terms in the Intensity equation
                # i.e. PositiveRe / PositiveIm ). Therefore we just grab the [0] element
                _amplitudes = {}
                for ampName, amplitudes in self.amplitudesMap.items():
                    _amplitudes[ampName] = self.daemonManager.gather_quantity("amplitude_value", amplitudes[0])
                nifty_amplitudes.append(_amplitudes)

                ## CALCULATE THE PHASE DIFFERENCES
                if self.phase_reference != "":
                    _phases = {}
                    for ax, pair in zip(phase_axes, pairs):
                        # Get the phase differences from the NIFTy fit
                        # Names like 'S0+' and 'D2++' are just the amplitude names
                        #  AmpTools requires additional information to locate the amplitude's values
                        #  The format should be REACTION::COHERENTSUM::AMPLITUDE. For the analysis
                        #  with Zlm amplitudes over the 4 polarized datasets, amplitudes exist in
                        #  different reactions and sums, but are all constrained to be the same.
                        #  We can just grab the first one.
                        _amp1, _amp2 = pair.split()
                        fullName_amp1 = self.amplitudesMap[_amp1][0]  # grabbin' the first
                        fullName_amp2 = self.amplitudesMap[_amp2][0]
                        fullname_pair = f"{fullName_amp1} {fullName_amp2}"

                        phaseDiffs = self.daemonManager.gather_quantity("phaseDiff", fullname_pair)
                        # if nearby points jumps more than 180 degrees, we need to correct for this
                        # phaseDiffs = np.unwrap( phaseDiffs )
                        # phaseDiffs -= circmean( phaseDiffs, low = -np.pi, high = np.pi )
                        # phaseDiffs -= np.mean( phaseDiffs )
                        phaseDiffs = np.rad2deg(phaseDiffs)
                        _phases[pair] = phaseDiffs
                    nifty_phases.append(_phases)

            ##############################################
            ####### INCLUDE NLL AT MEAN SAMPLE VALUE #####
            ##############################################
            sample_mean = np.mean(sampleValues, axis=0)
            sample_mean = sample_mean.reshape(-1, order="F")
            nll, _, _ = self.daemonManager.gather_likelihood(0, sample_mean)
            intensity_fig.suptitle(f"NLL @ Posterior Mean: {nll:.1f}", fontsize=20, c="black", y=0.95)

            ###########################
            ####### DRAW PLOTS ########
            ###########################

            def plot_ift(ax, samples, is_intensity=True, draw_bands_else_samples=False):
                linecolor = "tab:orange"
                mean, std = None, None

                _kde_kwargs = {"bw": 0.1, "nlevels": 20, "cmap": "viridis", "cbar": False}
                _kde_kwargs.update(self.kde_kwargs)

                if draw_bands_else_samples:  # aggregate to create bands
                    mean = np.mean(samples, axis=0)
                    std = np.std(samples, axis=0) if len(sampled_intensity) > 1 else np.zeros_like(mean)
                    ax.plot(masses, mean, c=linecolor, linestyle="--", zorder=ift_zorder, linewidth=3)
                    ax.fill_between(masses, mean - std, mean + std, alpha=0.4, color=linecolor, zorder=4)

                else:  # draw all samples as individual lines
                    _alpha = 0.3 if len(samples) > 1 else 1.0
                    _m = np.array([masses] * len(samples)).ravel()
                    _s = np.array(samples).ravel()
                    kde = KDE(np.vstack([_m, _s]).T)
                    for sample in samples:
                        # ax.plot(masses, sample, c=linecolor, linestyle='-', zorder=ift_zorder, linewidth=3, alpha=_alpha)
                        ax.scatter(masses, sample, color="blueviolet", s=10, zorder=3, marker="x", linewidth=1, alpha=0.8)
                        kde.add_curve(kde.scaler.transform(np.vstack([masses, sample]).T), is_intensity=is_intensity)
                    kde.plot(ax, is_intensity=is_intensity, **_kde_kwargs)

                return mean, std

            self.pkl_results = {"mass": masses}

            # 1. Draw intensity plots
            for ax, amp in zip(intensity_axes, partial_amp_names):
                sampled_intensity = [_intensity[amp] for _intensity in nifty_intensities]
                self.pkl_results[amp] = sampled_intensity  # save for diagnostics
                mean_intensities, std_intensities = plot_ift(ax, sampled_intensity, is_intensity=True, draw_bands_else_samples=self.draw_bands_else_samples)

                sampled_amplitudes = [_amplitude[amp] for _amplitude in nifty_amplitudes]
                self.pkl_results[amp + "_prod"] = sampled_amplitudes  # save for diagnostics

                # Compute goodness of fit metric comparing NIFTy and Generated intensities (if available)
                if self.gen_root_file != "" and self.gen_hist_fmt != "" and self.draw_bands_else_samples:
                    hist_name = self.gen_hist_fmt.format(amp)
                    gen_centers = self._gen[hist_name].axis().centers()
                    gen_counts = self._gen[hist_name].values() * self.gen_scale_factor
                    matched_idxs = np.searchsorted(gen_centers, masses)  # search indices of gen_centers that match masses
                    gof = self.goodness_of_fit(gen_counts[matched_idxs], mean_intensities, std_intensities)
                    metric = "$\chi^2$" if np.any(std_intensities) else "$R^2$"
                    ax.text(0.15, 0.9, f"{metric}: {gof:.2f}", transform=ax.transAxes, fontsize=20, c="black")

            # 2. Draw phase plots
            if self.phase_reference != "":
                for ax, pair in zip(phase_axes, pairs):
                    sampled_phaseDiff = [_phases[pair] for _phases in nifty_phases]
                    self.pkl_results[pair] = sampled_phaseDiff  # save for diagnostics
                    mean_phases, std_phases = plot_ift(ax, sampled_phaseDiff, is_intensity=False, draw_bands_else_samples=self.draw_bands_else_samples)

        ###########################
        ## Add labels and clean up
        ###########################

        [intensity_axes[i].set_xlabel(r"$M(\eta\pi)$ [GeV]", size=30) for i in range(nrows * ncols - ncols, nrows * ncols)]
        [intensity_axes[i].set_ylabel("Intensity", size=30) for i in range(0, nrows * ncols, ncols)]
        _ofile = self.fname_fmt.format(self.output_directory, "Intensity", global_iteration)
        intensity_fig.savefig(_ofile)
        os.system(f"ln -sfr {_ofile} {self.output_directory}/Intensity.{self.ext}")  # link the latest intensity fit result
        plt.close(intensity_fig)

        if self.phase_reference != "":
            [phase_axes[i].set_xlabel(r"$M(\eta\pi)$ [GeV]", size=30) for i in range(nrows * ncols - ncols, nrows * ncols)]
            [phase_axes[i].set_ylim(-180, 180) for i in range(nrows * ncols)]
            [phase_axes[i].set_ylabel("Phase [deg]", size=30) for i in range(0, nrows * ncols, ncols)]
            for pair, ax in zip(pairs, phase_axes):
                reference, amp = pair.split()
                label = f"$\phi$({prettyLabels[reference]},{prettyLabels[amp]})"
                color = "blue" if reference[-1] == "+" else "red"
                ax.text(0.1, 0.85, label, transform=ax.transAxes, fontsize=24, c=color, zorder=9)
            _ofile = self.fname_fmt.format(self.output_directory, "Phase", global_iteration)
            phase_fig.savefig(_ofile)
            os.system(f"ln -sfr {_ofile} {self.output_directory}/Phase.{self.ext}")  # link the latest phase fit result
            plt.close(phase_fig)

        print(f"DiagnosticCallback| Saved diagnostic plots for iteration: {global_iteration}")

    def create_gifs(self):
        print("DiagnosticCallback| Creating GIFs ... ", end="")

        cmd = "convert -delay {} -loop 0 {} {}"
        delay = 50  # ms
        base_d = self.output_directory

        # If saved as PDFs we will have to make a conversion first to png before GIFing
        if self.ext == "pdf":
            print("create_gifs| Converting PDFs to PNGs first ... ")
            for d in ["Intensity", "Phase"]:
                srcs = glob.glob(f"{base_d}/{d}_step*{self.ext}")
                for src in srcs:
                    base = os.path.basename(src)
                    os.system(f"pdftoppm -png {src} {base_d}/{base}")

        print("create_gifs| Creating GIFs ... ")
        for d in ["Intensity", "Phase"]:
            srcs = f"{base_d}/{d}_step*.png"
            dest = f"{base_d}/{d}.gif"
            os.system(cmd.format(delay, srcs, dest))

        # Clean up all the png intermediaries creating during GIFing process
        print("create_gifs| Cleaning up ... ")
        if self.ext == "pdf":
            for d in ["Intensity", "Phase"]:
                srcs = glob.glob(f"{base_d}/{d}_step*.png")
                for src in srcs:
                    os.system(f"rm {src}")

    def dump_final_samples(self):
        print("DiagnosticCallback| Dumping final samples to pkl ... ", end="")
        pkl_dump = f"{self.output_directory}/final_sample_intensities.pkl"
        with open(pkl_dump, "wb") as f:
            pkl.dump(self.pkl_results, f)
        print("Done!")

    @staticmethod
    def goodness_of_fit(fitted, expected, fitted_err):
        if np.all(fitted_err):  # if uncertainties are all non zeros, then use reduced chiSq metric
            return np.sum((fitted - expected) ** 2 / fitted_err**2) / (len(fitted) - 1)
        else:  # else use R2 metric
            return 1 - np.sum((fitted - expected) ** 2) / np.sum((expected - np.mean(expected)) ** 2)


class DaemonATI:
    """
    Single daemon process that loads a config file to sets up an AmpToolsInterface
    and awaits commands to compute NLL, gradient, and Hessians

    Args:
        cfgfile (str): path to an AmpTools' config file
        id (int): id of the daemon process -> index of cfgfile
        command_queue (multiprocessing.Queue): Queue to receive commands from
        nll_queue (multiprocessing.Queue): Queue to send results to
        quantity_queue (multiprocessing.Queue): Queue to send specific quantities to
    """

    def __init__(self, cfgfile, id, command_queue, nll_queue, quantity_queue, accelerator):
        #######################################################
        ################### SETUP AMPTOOLS ####################
        #######################################################

        USE_MPI, USE_GPU, RANK_MPI = atiSetup.setup(globals(), accelerator)

        assert not USE_MPI, "MPI not supported yet!"

        AmpToolsInterface.registerAmplitude(Zlm())
        AmpToolsInterface.registerAmplitude(Vec_ps_refl())
        AmpToolsInterface.registerAmplitude(OmegaDalitz())
        AmpToolsInterface.registerDataReader(DataReader())
        AmpToolsInterface.registerDataReader(DataReaderFilter())
        AmpToolsInterface.registerDataReader(DataReaderTEM())

        if USE_GPU:
            devs = GPUManager.getNumDevices()
            thisDevice = id % devs
            GPUManager.setThisDevice(thisDevice)

        # General bookkeeping
        self.command_queue = command_queue
        self.quantity_queue = quantity_queue
        self.nll_queue = nll_queue
        self.cfgfile = cfgfile
        self.id = id

        # Create AmpToolsInterface instance and support classes
        self.parser = ConfigFileParser(cfgfile)
        self.cfgInfo = self.parser.getConfigurationInfo()
        self.ati = AmpToolsInterface(self.cfgInfo)
        self.parMgr = LoadParameters(self.cfgInfo)

        # Get parameter names and initial values
        self.init_pars_value, self.keys, self.init_pars = self.parMgr.flatten_parameters()

        # This is the order AmpTools gradient vector returns with!
        parNameList = list(self.ati.parameterManager().getParNameList())
        if self.keys != parNameList:
            print("DaemonATI| The order of parameters do not match what AmpTools expects!", file=sys.stderr)
            for x, y in zip(self.keys, parNameList):
                print(f"  Passed / Expected : {x} / {y}", file=sys.stderr)
            raise ValueError("DaemonATI| Parameter order does not match what AmpTools (gradient calculator) expects!")

        self.npars = len(self.init_pars_value)
        self.parIdxMin = -1  # <- will be set by daemon_process wrapper
        self.parIdxMax = -1  # <- will be set by daemon_process wrapper

        # Computation functions
        self.Grad = nd.Gradient(self._NLL)
        self.Hessian = nd.Hessian(self._NLL)

        # Disable covariance updates <- Not using internal Minuit
        self.ati.parameterManager().setDoCovarianceUpdate(False)

        print("DaemonATI| Initialized ATI for config file:", cfgfile)

    def run(self):
        """
        Listen for command. "Full name" of amplitude ~ Reaction::Sum::Amplitude

        Expected Formats:
        * None                                  : exit command
        * ( max_order: int       , -1         ) : use initial parameter values found in cfg file
        * ( max_order: int       , List[pars] ) : list of parameter values to use
        * ( 'intensity'          , List[amps] ) : list of amplitudes to calculate intensity with
        * ( 'phaseDiff'          , str        ) : amplitude pair (full names) to calculate phase difference with
        * ( 'amplitudes'         , None       ) : return list of all amplitudes' full names ~ [ 'Reaction::Sum::Amp', ... ]
        * ( 'amplitude_value'    , str        ) : return complex value of amplitude (full name)
        * ( 'initial_par_values' , None       ) : return initial parameter values matching order of 'ordered_par_list'
        * ( 'ordered_par_list'   , None       ) : return list of all parameter names that AmpTools expects in order

        max_order = 0 -> compute NLL
        max_order = 1 -> compute NLL AND gradient
        max_order = 2 -> compute only Hessian

        Puts (nll, grad, hess) into output queue. Elements can take on None values if not requested
        """

        while True:  # Listen for commands
            command = self.command_queue.get()

            if command is None:
                break  # Exit signal

            arg1, arg2 = command

            if isinstance(arg1, int):  # Assume we want to compute NLL, grad, and(or) hess based on Flag
                max_order = arg1
                pars = arg2

                if isinstance(pars, int):  # assume its a flag
                    if pars == -1:  # testing flag <- use initial parameters
                        pars = self.init_pars_value
                else:  # assume its a list of parameters
                    pars = np.array(pars)
                    pars = pars[self.parIdxMin : self.parIdxMax]

                nll, grad, hess = None, None, None

                # Compute NLL, gradient, and Hessian
                if max_order in [0, 1]:
                    nll, grad = self.NLL(pars, max_order)
                if max_order == 2:
                    hess = self.Hessian(pars)

                # As results come in randomly, we need to track the id
                self.nll_queue.put((self.id, nll, grad, hess))

            else:  # assume we want to compute some quantity: intensity, phasediff, or amplitudes name list
                if arg1 == "intensity":
                    _amplitudes = arg2  # <- list of amplitudes to calculate intensity with (full names)
                    intensity = self.getIntensity(_amplitudes)
                    self.quantity_queue.put((self.id, intensity))

                elif arg1 == "phaseDiff":
                    _amplitude_pair = arg2  # <- amplitude pair to calculate phase difference with (full names)
                    phaseDiff = self.getPhaseDiff(_amplitude_pair)
                    self.quantity_queue.put((self.id, phaseDiff))

                elif arg1 == "amplitudes":
                    fullAmpNames = self.parMgr.allProdPars  # ~ [ 'Reaction::Sum::Amp', ... ]
                    self.quantity_queue.put((self.id, fullAmpNames))

                elif arg1 == "amplitude_value":
                    ampValue = complex(self.ati.parameterManager().findParameter(arg2).value())
                    self.quantity_queue.put((self.id, ampValue))

                elif arg1 == "initial_par_values":
                    self.quantity_queue.put((self.id, self.init_pars_value))

                elif arg1 == "ordered_par_list":
                    self.quantity_queue.put((self.id, self.keys))

                else:
                    raise ValueError("DaemonATI| Invalid argument passed to daemon process!")

    def NLL(self, par_values, max_order=0):
        """
        Request amptools to get the NLL, grad, hess (if requested) for a given set of parameters

        Args:
            par_values (List) of parameter values. Production coeffs are split into (re)al/(im)aginary parts

        Returns:
            nll (float): negative log-likelihood
            grad (List): gradient of nll
            hess (List): hessian of nll
        """

        #### Load parameters into AmpToolsInterface
        for name, value in zip(self.keys, par_values):
            # remove tag from name if multiple cfg files were passed
            self.ati.parameterManager()[name] = value

        # AmpTools returns 2NLL
        if max_order == 0:
            return 0.5 * self.ati.likelihood(), None
        else:  # any order >=1 we will return grad also
            nll, grad = self.ati.likelihoodAndGradient()
            return 0.5 * nll, [0.5 * g for g in grad]

    def _NLL(self, par_values):  # Used by numdifftools
        ## Load parameters into AmpToolsInterface
        for name, value in zip(self.keys, par_values):
            # remove tag from name if multiple cfg files were passed
            self.ati.parameterManager()[name] = value

        # AmpTools returns 2NLL
        return 0.5 * self.ati.likelihood()

    def getIntensity(self, amplitudes, accCorrected=True):
        """
        Calculate intensity given an amplitude list and AmpToolsInterface instance.
        Follows implementation in AmpTools' FitResults.cc
            intensity = sum_a sum_a' s_a s_a' V_a V*_a' NI( a, a' )

        Args:
            amplitudes (list): list of amplitudes to calculate intensity with
            accCorrected (bool): whether to use acceptance corrected intensities (or not)

        Returns:
            intensity (float): intensity
        """

        # TODO:
        # 1. Get scale factor between datasets

        ## AmpTools should update normalization integral interface every call to likelihood()
        ## We can then manually acess the interface and calculate the intensities manually.
        ## This only needs to be done for the final results so there is really no need to
        ## make it run faster.

        normIntMap = dict(self.ati.normIntMap())
        ati_parMgr = self.ati.parameterManager()

        intensity = 0

        for ampName in amplitudes:
            amp = ati_parMgr.findParameter(ampName).value()
            reaction = ampName.split("::")[0]

            for conjAmpName in amplitudes:
                conjAmp = ati_parMgr.findParameter(conjAmpName).value().conjugate()
                conjReaction = conjAmpName.split("::")[0]

                # Amps from different reactions are not coherent
                if reaction == conjReaction:
                    if accCorrected:
                        ampInt = normIntMap[reaction].ampInt(ampName, conjAmpName)
                    else:
                        ampInt = normIntMap[reaction].normInt(ampName, conjAmpName)
                else:
                    ampInt = complex(0, 0)

                # intensity += ( amp.real*conjAmp.real + amp.imag*conjAmp.imag ) * ampInt.real
                # intensity -= ( amp.imag*conjAmp.real - amp.real*conjAmp.imag ) * ampInt.imag
                intensity += (amp * conjAmp * ampInt).real

        return intensity

    def getPhaseDiff(self, amplitude_pair):
        """
        Calculate phase difference between an amplitude pair.

        Args:
            amplitude_pair (str): i.e. underscore separate amplitude pair (e.g. 'a1_a2'). Amplitudes must be full names!

        Returns:
            phaseDiff (float): phase difference
        """

        parMgr = self.ati.parameterManager()

        amp1, amp2 = amplitude_pair.split(" ")

        assert len(amp1.split("::")) == 3, "Amplitude (amp1) must be full name!"
        assert len(amp2.split("::")) == 3, "Amplitude (amp2) must be full name!"

        amp1 = parMgr.findParameter(amp1).value()
        amp2 = parMgr.findParameter(amp2).value()

        # np.angle returns on [-pi, pi]
        # phase diff would therefore return on [-2pi, 2pi]
        # This is the same format that AmpTools.FitResults.phaseDiff function uses
        phase_diff = np.angle(amp1) - np.angle(amp2)

        # put it back on [-pi, pi] range
        return np.arctan2(np.sin(phase_diff), np.cos(phase_diff))


def daemon_process(
    cfgfile,
    id,
    command_queue,
    nll_queue,
    quantity_queue,
    accelerator,
    shared_param_dict,
    lock,
):
    """
    Wrapper function to spawn daemon processes

    Args:
        See Args for DaemonATI
        shared_param_dict (multiprocessing.Manager.dict): dictionary to store info on how to partition parameters array
    """

    print("daemon_process| Spawning daemon for:", cfgfile)
    daemon = DaemonATI(cfgfile, id, command_queue, nll_queue, quantity_queue, accelerator)

    # with lock:
    #     if 'partitions' not in shared_param_dict:
    #         shared_param_dict['partitions'] = [ 0 ]
    #          # have to recreate and add (+=), can't just append
    #     shared_param_dict['partitions'] += [ daemon.npars + shared_param_dict['partitions'][-1] ]
    #     print(f"daemon_process| {cfgfile} Parameter partition: {shared_param_dict['partitions']}")
    #     daemon.parIdxMin = shared_param_dict['partitions'][-2]
    #     daemon.parIdxMax = shared_param_dict['partitions'][-1]

    # order_queue.pop(0)

    daemon.parIdxMin = id * daemon.npars
    daemon.parIdxMax = (id + 1) * daemon.npars

    daemon.run()


class DaemonManager:
    def __init__(self, cfgfiles, accelerator):
        """
        Manager class to spawn daemon processes and send commands to them

        Args:
            cfgfiles (List[str]): list of paths to AmpTools config files
        """

        self.manager = multiprocessing.Manager()
        self.accelerator = accelerator
        self.shared_param_dict = self.manager.dict()

        self.cfgfiles = cfgfiles
        self.ndaemons = len(cfgfiles)
        self.daemons = []
        self.queues = []  # <- Send new parameters to daemons
        self.nll_queue = multiprocessing.Queue()  # <- receive (nll, grad, hess) from daemons
        self.quantity_queue = multiprocessing.Queue()  # <- receive a specific quantity from daemons

        print("DaemonManager| Initialized")

    def spawn_daemons(self):
        self.lock = multiprocessing.Lock()

        # Create Queues to send data to daemon processes
        for _ in range(self.ndaemons):
            self.queues.append(multiprocessing.Queue())

        # Start Daemon processes
        #   Currently, NIFTy model expects a single large parameter vector merging across mass bins
        for id, queue, cfgfile in zip(range(len(self.queues)), self.queues, self.cfgfiles):
            d_process = multiprocessing.Process(target=daemon_process, args=(cfgfile, id, queue, self.nll_queue, self.quantity_queue, self.accelerator, self.shared_param_dict, self.lock))
            d_process.daemon = True
            d_process.start()
            self.daemons.append(d_process)

        print("DaemonManager| Spawned daemons")

    def gather_likelihood(self, order, par_values):
        # print(f'DaemonManager| Gathering results with deriv order: {order}')

        for queue in self.queues:
            queue.put((order, par_values))

        # Await results
        results = [self.nll_queue.get() for _ in range(self.ndaemons)]
        results.sort(key=lambda x: x[0])  # sort by id <- cfgfile order

        nll, grad, hess = None, None, None

        if order in [0, 1]:
            nll = [result[1] for result in results]
            nll = np.sum(nll)
        if order == 1:
            grad = [result[2] for result in results]
            grad = np.concatenate(grad)
        if order == 2:
            hess = [result[3] for result in results]

        return nll, grad, hess

    def gather_quantity(self, quantity, arg=None):
        """
        Gather a specific quantity from EVERY daemon. See DaemonATI.run(...) for expected formats

        Args:
            quantity (str): quantity to gather
            arg (List): Artifact for gather_likelihood. No current use in gather_quanitity but may be useful in the future
        """

        _expected_terms = ["intensity", "phaseDiff", "amplitudes", "amplitude_value", "initial_par_values", "ordered_par_list"]
        assert quantity in _expected_terms, "DaemonManager| Invalid quantity!"
        # print(f'DaemonManager| Gathering {quantity}')

        # Send request for result to each daemon
        for queue in self.queues:
            queue.put((quantity, arg))

        # Await results
        results = [self.quantity_queue.get() for _ in range(self.ndaemons)]
        results.sort(key=lambda x: x[0])

        quantities = [result[1] for result in results]

        return quantities

    def despawn_daemons(self):
        # Send None to indicate completion, daemons no longer listening for commands
        for queue in self.queues:
            queue.put(None)

        # Wait for the daemon process to finish
        for d_process in self.daemons:
            d_process.join()

        print("DaemonManager| Despawned daemons")
