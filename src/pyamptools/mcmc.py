import numpy as np
import pandas as pd
import emcee
import os
import matplotlib.pyplot as plt
import corner
import argparse
import time
import sys
import yaml
from pyamptools.utility.load_parameters import LoadParameters, createMovesMixtureFromDict
from pyamptools.utility.general import glob_sort_captured, safe_getsize


class mcmcManager:
    """
    This class manages the MCMC sampling process and the interaction with AmpToolsInterface instances
    """

    def __init__(self, atis, LoadParametersSamplers, ofile):
        self.ofile = ofile
        self.finishedSetup = False
        self.samples = None

        ## These three attributes are lists of the same size, one for each amptools config file
        ## NOTE: WE ASSUME THAT THE AMPLITUDES ARE THE SAME ACROSS ALL CONFIG FILES
        self.atis = atis
        self.LoadParametersSamplers = LoadParametersSamplers

    def Prior(self, par_values, keys):
        """
        (Log of) Prior distribution for the par_values

        Args:
            par_values (MinuitParameter*): list of parameter names and values. Production coeffs are split into (re)al/(im)aginary parts
            keys (str): parameter names
                If multiple config files were passed, the parameter names will be appended with a '_i' tag where i is the ith cfg file

        Returns:
            prior (float): Log prior probability
        """
        return 0

    def Likelihood(self, ati):
        """Returns Log likelihood from AmpToolsInterface instance. Separation for profiling"""
        return -ati.likelihood()

    def LogProb(
        self,
        par_values,
        keys,
    ):
        """
        Definition of the (Log) Posterior distribution

        Args:
            par_values (MinuitParameter*): list of parameter names and values. Production coeffs are split into (re)al/(im)aginary parts
            keys (str): parameter names
                If multiple config files were passed, the parameter names will be appended with a '_i' tag where i is the ith cfg file

        Returns:
            log_prob (float): Log posterior probability
        """

        ## Calculate Log likelihood
        log_prob = 0
        n_atis = len(self.atis)
        for i, ati in zip(range(n_atis), self.atis):
            _par_values = par_values[self.paramPartitions[i] : self.paramPartitions[i + 1]]
            _keys = keys[self.paramPartitions[i] : self.paramPartitions[i + 1]]
            for name, value in zip(_keys, _par_values):
                # remove tag from name if multiple cfg files were passed
                name = name[: name.rfind("_")] if n_atis > 1 else name
                ati.parameterManager()[name] = value

        log_prob += self.Likelihood(ati) + self.Prior(par_values, keys)

        return log_prob

    def perform_mcmc(
        self,
        burnIn: int = 100,
        nwalkers: int = 32,
        nsamples: int = 1000,
        intensity_dump: str = "",
        ### Non-CLI arguments Below ###
        params_dict: dict = {},
        moves_mixture=[emcee.moves.StretchMove()],
        sampler_kwargs={},
    ):
        """
        Performs MCMC sampling

        Args:
            burnIn (int): number of burn-in steps
            nwalkers (int): number of walkers
            nsamples (int): number of samples
            intensity_dump (str): file to dump intensities map into (numpy feather file output)
            params_dict (dict): Dictionary of parameter names (complex production coeffs and amplitude params) and values.
            moves_mixture [ (emcee.moves.{move}(kwargs), probability) ]: List of tuples of moves (with kwargs pluggin in) and their probability
            sampler_kwargs (dict): Additional keyword arguments to pass to emcee.EnsembleSampler()

        Returns:
            results (dict): Dictionary containing results of MCMC sampling (samples, MAP, acceptance fraction, autocorrelation time, etc.)
        """

        print("\n ================== PERFORM_MCMC() KWARGS CONFIGURATION ================== ")
        print(f"nwalkers: {nwalkers}")
        print(f"burnIn: {burnIn}")
        print(f"nsamples: {nsamples}")
        print(f"params_dict: {params_dict}")
        print(f"moves_mixture: {moves_mixture}")
        print(f"sampler_kwargs: {sampler_kwargs}")
        print(" ========================================================================= \n")

        ########## SETUP PARAMETERS ############
        ## AmpTools config files can be initialized to MLE values
        ##   this can be acheived by running a MLE fit with --seedfile
        ##   and using the include directive in the config file to add the seedfile
        ##   to override values
        ## doNotScatter = True  will initialize walkers in an N-ball around the initial values
        ## doNotScatter = False will uniformly sample (or scatter) on user-specified interval
        params = {}
        doNotScatter = True
        if len(params_dict) != 0:
            first_element = next(iter(params_dict.values()))
            doNotScatter = not isinstance(first_element, list) or len(params_dict) == 0
            params = params_dict if doNotScatter else {k: v[0] for k, v in params_dict.items()}

        ####### LOAD PARAMETERS #####
        initial_values, keys, par_names_flat = [], [], []
        paramPartitions = [0]
        for i, LoadParametersSampler in enumerate(self.LoadParametersSamplers):
            values, ks, pars = LoadParametersSampler.flatten_parameters(params)
            initial_values.extend(values)
            tag = f"_{i}" if len(self.LoadParametersSamplers) > 1 else ""
            keys.extend([f"{k}{tag}" for k in ks])
            par_names_flat.extend([f"{par}{tag}" for par in pars])
            paramPartitions.append(len(values) + paramPartitions[-1])

        ####### CHECK FOR MINIMUM WALKERS ########
        if nwalkers < 2 * len(initial_values):
            print(f"\n[emcee req.] Number of walkers must be at least twice the number of \
                  parameters. Overwriting nwalkers to 2*len(initial_values) = 2*{len(initial_values)}\n")
            nwalkers = 2 * len(initial_values)

        nDim = len(initial_values)

        ################### TRACK SOME ATTRIBUTES ###################
        self.nwalkers = nwalkers
        self.paramPartitions = paramPartitions  # list of indices to re-group par_names_flat into each cfg file
        self.par_names_flat = par_names_flat
        self.keys = keys
        self.initial_values = initial_values
        self.amplitudeMap = self.getAmplitudeMaps()

        ############## RUN MCMC IF RESULTS DOES NOT ALREADY EXIST ##############
        fit_start_time = time.time()
        output_file = f"{self.ofile}"
        fileTooSmall = safe_getsize(output_file) < 1e4  # remake if too small, likely corrupted initialization
        if not os.path.exists(output_file) or fileTooSmall:
            if fileTooSmall:
                os.system(f"rm -f {output_file}")

            print(" ================== RUNNING MCMC ================== ")
            if doNotScatter:
                print("Initializing walkers in an N-ball around the initial estimate")
                par_values = np.array(initial_values)
                par_values = np.repeat(par_values, nwalkers).reshape(nDim, nwalkers).T
                par_values[par_values == 0] += 1e-2  # avoid 0 values, leads to large condition number for sampler
                par_values *= 1 + 0.01 * np.random.normal(0, 1, size=(nwalkers, nDim))
            else:
                print("Randomly sampling on user-specified interval")
                par_values = np.empty((nwalkers, nDim))
                for k, (_, mini, maxi) in params_dict.items():
                    par_values[:, keys.index(k)] = np.random.uniform(mini, maxi, size=nwalkers)

            backend = emcee.backends.HDFBackend(output_file)
            backend.reset(nwalkers, nDim)
            sampler = emcee.EnsembleSampler(nwalkers, nDim, self.LogProb, args=[keys], moves=moves_mixture, backend=backend)

            _sampler_kwargs = {"progress": True}
            _sampler_kwargs.update(sampler_kwargs)
            if burnIn != 0:
                print("\n[Burn-in beginning]\n")
                state = sampler.run_mcmc(par_values, burnIn)
                sampler.reset()
                print("\n[Burn-in complete. Running sampler]\n")
                sampler.run_mcmc(state, nsamples, **_sampler_kwargs)
            else:
                sampler.run_mcmc(par_values, nsamples, **_sampler_kwargs)

            self.samples = sampler.get_chain(flat=True)
            self.intensities = self.getIntensity(self.amplitudeMap, accCorrected=True)
            print("\n[Sampler complete]\n")

        else:  # Found already existing HDF5 file, will just load it
            print(" ================== LOADING MCMC ================== ")
            sampler = emcee.backends.HDFBackend(output_file)
            self.samples = sampler.get_chain(flat=True)
            if not os.path.exists(intensity_dump):
                self.intensities = None
            else:
                self.intensities = pd.read_feather(intensity_dump).values
                self.intensities = self.intensities.reshape((len(self.atis), nsamples * nwalkers, len(self.amplitudeMap)))

        fit_end_time = time.time()

        ################# CALCULATE AUTOCORR AND ACCEPTANCE FRACTION #################
        if isinstance(sampler, emcee.EnsembleSampler):
            acceptance_fraction = np.mean(sampler.acceptance_fraction)
        else:  # HDF5 backend
            # function implementation in github, acceptance_fraction not available from HDF5 backend
            acceptance_fraction = np.mean(sampler.accepted / sampler.iteration)
        autocorr_time = np.mean(sampler.get_autocorr_time(quiet=True))
        print(f"Mean acceptance fraction: {acceptance_fraction:.3f}")
        print(f"Autocorrelation time: {autocorr_time:.3f} steps")

        ################### COMPUTE MAP ESTIMATE ###################
        map_idx = np.argmax(sampler.get_log_prob(flat=True))  # maximum a posteriori (MAP) location
        map_estimate = self.samples[map_idx, :]
        median_estimate = np.median(self.samples, axis=0)
        upper_estimate = np.percentile(self.samples, 84, axis=0)
        lower_estimate = np.percentile(self.samples, 16, axis=0)
        mean_estimate = np.mean(self.samples, axis=0)
        std_estimate = np.std(self.samples, axis=0)

        stats_dump = "\n# ====================================== RESULTS ====================================== \n#\n"
        stats_dump += f"# Estimates from {len(self.samples)} samples obtained over {nwalkers} walkers:\n"
        stats_dump += f'#   {"":20}   {"MAP":^10} {"16%":^10} {"50%":^10} {"84%":^10}\n'
        csv_dump = "\n# =========================== CSV OUTPUT ===========================\n\n"
        csv_dump += "var_name, map, lower, median, upper, mean, std\n"
        for name, mapv, lowv, medv, upv, meanv, stdv in zip(par_names_flat, map_estimate, lower_estimate, median_estimate, upper_estimate, mean_estimate, std_estimate):
            stats_dump += f"#   {name:<20} = {mapv:<10.3f} {lowv:<10.3f} {medv:<10.3f} {upv:<10.3f} {meanv:<10.3f} {stdv:<10.3f}\n"
            csv_dump += f"{name}, {mapv:0.3f}, {lowv:0.3f}, {medv:0.3f}, {upv:0.3f}, {meanv:0.3f}, {stdv:0.3f}\n"
        stats_dump += "# ======================================================================================\n"
        with open(f"{os.path.dirname(self.ofile)}/results_stats.txt", "w") as f:
            f.write(stats_dump)
            f.write(csv_dump)
        print(stats_dump)

        ################### TRACK SOME ADDITIONAL ATTRIBUTES ###################
        self.map_estimate = map_estimate
        self.map_sample_idx = map_idx
        self.elapsed_fit_time = fit_end_time - fit_start_time

        ################### DUMP INTENSITIES IF REQUESTED ###################
        # Enough information in the intensites array to calculate fit fractions
        if intensity_dump != "":
            if "." not in intensity_dump:
                intensity_dump += ".feather"
            columns = list(self.amplitudeMap.keys())
            if len(self.atis) > 1:
                columns = [f"{ampName}_{i}" for i in range(len(self.atis)) for ampName in columns]
            dump = np.concatenate(self.intensities, axis=1)
            dump = pd.DataFrame(dump, columns=columns)
            dump.to_feather(intensity_dump)
            print(f"\nMCMC intensity samples saved to {intensity_dump}")
            print(f"  shape: {dump.shape}\n")

    def getIntensity(self, amplitudeMap, accCorrected=True):
        """
        Get intensity from a dictionary of a list of amplitudes

        Args:
            amplitudeMap: see getIntensity_singleATI()
            accCorrected (bool): whether to use acceptance corrected intensities (or not)

        Returns:
            intensities (List[array]): List of intensity arrays for each ATI instance. Shape ~ [ati, samples, amplitudes]
        """
        intensities = []
        for ati_index in range(len(self.atis)):
            intensities.append(self.getIntensity_singleATI(ati_index, amplitudeMap, accCorrected))
        return intensities

    def getIntensity_singleATI(self, ati_index, amplitudeMap, accCorrected=True):
        """
        Calculate intensity given an amplitude list and AmpToolsInterface instance.
        Follows implementation in AmpTools' FitResults.intensity = sum_a sum_a' s_a s_a' V_a V*_a' NI( a, a' )

        Args:
            ati_index (int): index of AmpToolsInterface instance to use
            amplitudeMap (dict[str, list]): Dictionary to calculate intensity with. Keys are amp names, values are lists of amplitudes belonging to that amp. Set to None to use the predetermined amplitudeMap attribute
            accCorrected (bool): whether to use acceptance corrected intensities (or not)

        Returns:
            intensity (np.array): intensity of coherent sum of amplitudes (defined by amplitudeMap) across mcmc samples. Shape ~ [samples, amplitudes]
        """

        print("\nCalculating intensity with amplitudes dictionary:")
        for k, v in amplitudeMap.items():
            print(f"  {k} -> {v}")

        # TODO:
        # 1. Get scale factor between datasets

        ## AmpTools should update normalization integral interface every call to likelihood()
        ## NOTE! The normalization integrals is the same between fit iterations if amplitudes
        ##       do not have any free parameters.
        ##       WE MAKE THE ASSUMPTION THAT THE NORMALIZATION INTEGRALS ARE CONSTANT TO
        ##       SIMPLIFY THE CALCULATIONS
        ##       We reload the params and use the ATI manager to handle constraints, etc

        if self.samples is None:
            print("No samples to calculate intensity. Run perform_mcmc() first")
            return

        ## LOOP OVER ATI INSTANCES (cfg files) to create a list of structured intensity arrays
        # where the columns are the keys of amplitudeMap and the rows are the samples
        intensities = np.empty((len(self.samples), len(amplitudeMap)))

        for i, (ampName, amplitudes) in enumerate(amplitudeMap.items()):
            normIntMap = dict(self.atis[ati_index].normIntMap())
            ati_parMgr = self.atis[ati_index].parameterManager()

            ## FOR ALL SAMPLES, RELOAD VALUES
            samples_intensity = np.empty(len(self.samples))
            for j in range(len(self.samples)):
                _par_values = self.samples[j, self.paramPartitions[ati_index] : self.paramPartitions[ati_index + 1]]
                _keys = self.keys[self.paramPartitions[ati_index] : self.paramPartitions[ati_index + 1]]

                for name, value in zip(_keys, _par_values):
                    # remove tag from name if multiple cfg files were passed
                    name = name[: name.rfind("_")] if len(self.atis) > 1 else name
                    ati_parMgr[name] = value

                ## CALCULATE INTENSITY WITH GIVEN AMPLITUDE LIST
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
                                # TODO: AmpTools does not like touching the generated MC (used for
                                # acceptance correction) when it thinks it is fitting, because generally
                                # this is not needed. We force it to do so by passing forceUseCache = True to ampInt
                                ampInt = normIntMap[reaction].ampInt(ampName, conjAmpName, True)
                            else:
                                ampInt = normIntMap[reaction].normInt(ampName, conjAmpName)
                        else:
                            ampInt = complex(0, 0)

                        intensity += (amp * conjAmp * ampInt).real

                samples_intensity[j] = intensity

            intensities[:, i] = samples_intensity

        return intensities

    def getAmplitudeMaps(self):
        """
        Each amplitude map is a dictionary for a UNIQUE amplitude name matching to a list of full name amptools amplitudes
        This can be used to calculate the intensity for each amplitude integrating over some contributions
        Example key, value pair: Sp0+ -> [reaction_pol000::PositiveRe::Sp0+, reaction_pol000::PositiveIm::Sp0+, reaction_pol045::PositiveRe::Sp0+, ... ]
        """
        parMgr = self.LoadParametersSamplers[0]
        fullAmpNames = parMgr.allProdPars
        amplitudeMap = {}
        for fullAmpName in fullAmpNames:
            ampName = fullAmpName.split("::")[-1]
            if ampName not in amplitudeMap:
                amplitudeMap[ampName] = [fullAmpName]
            else:
                amplitudeMap[ampName].append(fullAmpName)

        amplitudeMap["total"] = fullAmpNames

        return amplitudeMap

    def draw_corner(
        self,
        corner_ofile="",
        format="cartesian",
        save=True,
        kwargs={},
    ):
        """
        Draws a corner plot to visualize sampling and parameter correlations

        Args:
            corner_ofile (str): Output file name
            format (str): plot real/imag parts (cartesian), intensities, or fit fractions. {cartesian, intensity, fitfrac}
            save (bool): Save the figure to a file or return figure. Default True = Save
            kwargs (dict): Keyword arguments to pass to corner.corner()

        Returns:
            fig (matplotlib.figure.Figure): Figure object if save=False
        """

        format = format.lower()
        assert format in ["cartesian", "intensity", "fitfrac"], f"Invalid format {format}. Must be one of [cartesian, intensity, fitfrac]"

        if corner_ofile == "":
            print("No corner plot output file specified. Not drawing corner plot")
            return

        if self.samples is None:
            print("No samples to draw corner plot. Run perform_mcmc() first")
            return

        if self.intensities is None and format != "cartesian":
            print("No intensity samples to draw corner plot")
            return

        def plot_axvh_fit_params(
            axes,  # axes object
            values,  # [real1, imag1, real2, imag2, ...]
            nDim,  # number of parameters
            color="black",
        ):
            """Draws lines on the corner plot to indicate the specific parameter values"""
            for yi in range(nDim):
                for xi in range(yi):
                    axes[yi, xi].axvline(values[xi], color=color, linewidth=4, alpha=0.8)
                    axes[yi, xi].axhline(values[yi], color=color, linewidth=4, alpha=0.8)
                axes[yi, yi].axvline(values[yi], color=color, linewidth=4, alpha=0.8)

        ####### DRAW CORNER PLOT #######
        corner_kwargs = {"color": "black", "show_titles": True}
        corner_kwargs.update(kwargs)

        natis = len(self.atis)
        if format == "cartesian":
            if natis == 1:
                labels = [f'{name.split("::")[-1]}' for name in self.par_names_flat]
            else:
                labels = [f'{name.split("::")[-1]}_{iati}' for iati in range(natis) for name in self.par_names_flat]
            samples = self.samples
        else:
            amplitudeMap = self.amplitudeMap
            intensities = self.intensities

            if format == "fitfrac":
                for i in range(len(intensities)):
                    intensities[i] /= intensities[i][:, -1][:, None]  # divide each amplitude column by total intensity
            samples = np.concatenate([intensity[:, :-1] for intensity in intensities], axis=1)

            tag = "FF" if format == "fitfrac" else "I"
            if natis == 1:
                labels = [f"{tag}[{ampName}]" for ampName in list(amplitudeMap.keys())[:-1]]
            else:
                labels = [f"{tag}[{ampName}]_{iati}" for iati in range(natis) for ampName in list(amplitudeMap.keys())[:-1]]  # ignore last key is 'total'

            assert samples.shape[1] == len(labels), f"Cols in samples must match labels length. Got {samples.shape[1]} cols, {len(labels)} labels"

        fig = corner.corner(samples, labels=labels, **corner_kwargs)

        nDim = samples.shape[1]
        axes = np.array(fig.axes).reshape((nDim, nDim))
        plot_axvh_fit_params(axes, samples[self.map_sample_idx], nDim, color="royalblue")
        if format == "cartesian":
            plot_axvh_fit_params(axes, self.initial_values, nDim, color="tab:green")
        else:
            print("Not overlaying initial values, have not implemented conversion from Real/Imag parts to intensities or fit fractions")

        if save:
            plt.savefig(f"{corner_ofile}")
        else:
            return fig


def _cli_mcmc():
    """Command line interface for performing mcmc fits"""

    # NOTE: YAML file can be supplied to override the command line args and is
    # the only way to construct different emcee moves and initialization of parameters
    # as this process can be quite complex and is not easily done through the CLI.
    # Without YAML override, emcee will be initialized to a N-ball around the
    # amptools cfg file's values

    start_time = time.time()

    parser = argparse.ArgumentParser(description="emcee fitter")
    parser.add_argument("cfgfiles", type=str, help="Config file name or glob pattern")
    parser.add_argument("-nw", "--nwalkers", type=int, default=32, help="Number of walkers. Default 32")
    parser.add_argument("-b", "--burnin", type=int, default=100, help="Number of burn-in steps. Default 100")
    parser.add_argument("-n", "--nsamples", type=int, default=1000, help="Number of samples. Default 1000")
    parser.add_argument("-w", "--overwrite", action="store_true", help="Overwrite existing output file if exists")
    parser.add_argument("-a", "--accelerator", type=str, default="mpigpu", help="Use accelerator if available ~ [cpu, gpu, mpi, mpigpu, gpumpi]")
    parser.add_argument("-o", "--ofile", type=str, default="mcmc/emcee_state.h5", help='Output file name. Default "mcmc/emcee_state.h5"')
    parser.add_argument("-i", "--intensity_dump", type=str, default="mcmc/samples_intensity.feather", help='Output file name for intensity dump. Default "mcmc/samples_intensity.feather". "" = do not dump')
    parser.add_argument("-c", "--corner_ofile", type=str, default="", help="Corner plot output file name. Default empty str = do not draw")
    parser.add_argument("-cf", "--corner_format", type=str, default="cartesian", help='Corner plot format. Default "cartesian" to plot Real/Imag parts. Options: [cartesian, intensity, fitfrac]')
    parser.add_argument("-s", "--seed", type=int, default=42, help="RNG seed for consistent walker random initialization. Default 42")
    parser.add_argument("-y", "--yaml_override", type=str, default="", help='Path to YAML file containing parameter overrides. Default "" = no override')

    args = parser.parse_args(sys.argv[1:])

    if args.yaml_override == "":
        yaml_override = {}
    if args.yaml_override != "":
        with open(args.yaml_override, "r") as f:
            yaml_override = yaml.safe_load(f)

    ################ Override CLI arguments with YAML file if specified ###################
    cfgfiles = args.cfgfiles if "cfgfiles" not in yaml_override else yaml_override["cfgfiles"]
    ofile = args.ofile if "ofile" not in yaml_override else yaml_override["ofile"]
    corner_ofile = args.corner_ofile if "corner_ofile" not in yaml_override else yaml_override["corner_ofile"]
    corner_format = args.corner_format if "corner_format" not in yaml_override else yaml_override["corner_format"]
    nwalkers = args.nwalkers if "nwalkers" not in yaml_override else yaml_override["nwalkers"]
    burnIn = args.burnin if "burnin" not in yaml_override else yaml_override["burnin"]
    nsamples = args.nsamples if "nsamples" not in yaml_override else yaml_override["nsamples"]
    intensity_dump = args.intensity_dump if "intensity_dump" not in yaml_override else yaml_override["intensity_dump"]
    overwrite_ofile = args.overwrite if "overwrite" not in yaml_override else yaml_override["overwrite"]
    seed = args.seed if "seed" not in yaml_override else yaml_override["seed"]
    params_dict = {} if "params_dict" not in yaml_override else yaml_override["params_dict"]
    moves_mixture = [emcee.moves.StretchMove()] if "moves_mixture" not in yaml_override else createMovesMixtureFromDict(yaml_override["moves_mixture"])

    cfgfiles = glob_sort_captured(cfgfiles)  # list of sorted config files based on captured number
    for cfgfile in cfgfiles:
        assert os.path.exists(cfgfile), "Config file does not exist at specified path"

    ################### LOAD LIBRARIES ##################
    from pyamptools import atiSetup

    USE_MPI, USE_GPU, RANK_MPI = atiSetup.setup(globals(), args.accelerator)

    ############## LOAD CONFIGURATION FILE #############
    parsers = [ConfigFileParser(cfgfile) for cfgfile in cfgfiles]
    cfgInfos = [parser.getConfigurationInfo() for parser in parsers]  # List of ConfigurationInfo

    ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude(Zlm())
    AmpToolsInterface.registerAmplitude(Vec_ps_refl())
    AmpToolsInterface.registerAmplitude(OmegaDalitz())
    AmpToolsInterface.registerAmplitude(BreitWigner())
    AmpToolsInterface.registerAmplitude(Piecewise())
    AmpToolsInterface.registerAmplitude(PhaseOffset())
    AmpToolsInterface.registerAmplitude(TwoPiAngles())
    AmpToolsInterface.registerDataReader(DataReader())
    AmpToolsInterface.registerDataReader(DataReaderTEM())
    AmpToolsInterface.registerDataReader(DataReaderFilter())

    atis = [AmpToolsInterface(cfgInfo) for cfgInfo in cfgInfos]
    LoadParametersSamplers = [LoadParameters(cfgInfo) for cfgInfo in cfgInfos]
    [ati.parameterManager().setDoCovarianceUpdate(False) for ati in atis]  # No internal Minuit fit = no covariance matrix

    ############## RUN MCMC ##############
    np.random.seed(seed)

    if RANK_MPI == 0:
        print("\n ====================================================================================")
        print(f" cfgfiles: {cfgfiles}")
        print(f" ofile: {ofile}")
        print(f" corner_ofile: {corner_ofile}")
        print(f" corner_format: {corner_format}")
        print(f" nwalkers: {nwalkers}")
        print(f" burnIn: {burnIn}")
        print(f" nsamples: {nsamples}")
        print(f" intensity_dump: {intensity_dump}")
        print(f" overwrite_ofile: {overwrite_ofile}")
        print(f" seed: {seed}")
        print(f" params_dict: {params_dict}")
        print(f" moves_mixture: {moves_mixture}")
        print(" ====================================================================================\n")

        print("\n===============================")
        print(f"Loaded {len(cfgInfos)} config files")
        print("===============================\n")
        displayN = 1  # Modify me to show display more cfg files. Very verbose!
        for cfgfile, cfgInfo in zip(cfgfiles[:displayN], cfgInfos[:displayN]):
            print("\n=============================================")
            print(f"Display Of This Config File Below: {cfgfile}")
            print("=============================================\n")
            cfgInfo.display()

        ############## PREPARE FOR SAMPLER ##############
        if os.path.isfile(f"{ofile}") and overwrite_ofile:
            os.system(f"rm -f {ofile}")
            print("Overwriting existing output file!")
        if "/" in ofile:
            ofolder = ofile[: ofile.rfind("/")]
            os.system(f"mkdir -p {ofolder}")

        mcmcMgr = mcmcManager(atis, LoadParametersSamplers, ofile)

        mcmcMgr.perform_mcmc(
            nwalkers=nwalkers,
            burnIn=burnIn,
            nsamples=nsamples,
            intensity_dump=intensity_dump,
            params_dict=params_dict,
            moves_mixture=moves_mixture,
        )

        mcmcMgr.draw_corner(f"{corner_ofile}", format=corner_format)

        print(f"Fit time: {mcmcMgr.elapsed_fit_time} seconds")
        print(f"Total time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    _cli_mcmc()
