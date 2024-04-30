import os
import argparse
import time
import sys
from pyamptools.utility.load_parameters import LoadParameters
from pyamptools.mcmc import mcmcManager
import optuna
import logging
import numpy as np
from emcee.moves import StretchMove, WalkMove, DEMove, DESnookerMove
import yaml
from pyamptools import atiSetup

##############################################
# TODO: Likely not complete, fix later
##############################################


class Objective:
    """
    Modified objective function (Class) for Optuna that now takes an initialization list
    """

    def __init__(self, conditions):
        """Primes the objective function with the initialization list"""
        for k, v in conditions.items():
            setattr(self, k, v)

    def __call__(self, trial):
        """
        Performs a single trial of the objective function, sampling a mixture of moves.
            ATM, Contains hard-coded parameter ranges and walker initialization ranges for a specific test case

        Args:
            trial (optuna.trial.Trial): Optuna trial object
        """

        ############## PREPARE FOR SAMPLER ##############
        ofile = f"mcmc_{trial.number}.h5"
        corner_ofile = f"corner_{trial.number}.png"
        if os.path.isfile(f"{ofolder}/{ofile}") and self.overwrite_ofile:
            os.system(f"rm {ofolder}/{ofile}")
            print("Overwriting existing output file!")

        values, keys, _ = LoadParametersSampler[0].flatten_parameters()
        nFreePars = len(values)

        ########### ASK OPTUNA FOR MOVES AND ASSOCIATED PROBABILITIES ###########
        deMove_s = trial.suggest_float("deMove_s", 1e-6, 1e-4)
        deMove_prob = trial.suggest_float("deMove_prob", 0.0, 1.0)
        stretchMove_a = trial.suggest_float("stretchMove_a", 1.0, 3.0)
        stretchMove_prob = trial.suggest_float("stretchMove_prob", 0.0, 1.0)
        deSnookerMove_gs = trial.suggest_float("deSnookerMove_gs", 1.2, 2.5)
        deSnookerMove_prob = trial.suggest_float("deSnookerMove_prob", 0.0, 1.0)
        walkMove_prob = trial.suggest_float("walkMove_prob", 0.0, 1.0)
        walkMove_s = trial.suggest_int("walkMove_s", 2, nFreePars)  # >2 as it calculates std/covs

        deMove = DEMove(deMove_s)
        stretchMove = StretchMove(stretchMove_a)
        deSnookerMove = DESnookerMove(deSnookerMove_gs)
        walkMove = WalkMove(walkMove_s)

        # Weights will get normalized by emcee
        moves_mixture = [
            (stretchMove, stretchMove_prob),
            (deMove, deMove_prob),
            (deSnookerMove, deSnookerMove_prob),
            (walkMove, walkMove_prob),
        ]
        moves_dict = {  # For reproducibility we dump this to a YAML file
            "StretchMove": {"kwargs": {"a": stretchMove_a}, "prob": stretchMove_prob},
            "DEMove": {"kwargs": {"sigma": deMove_s}, "prob": deMove_prob},
            "DESnookerMove": {"kwargs": {"gammas": deSnookerMove_gs}, "prob": deSnookerMove_prob},
            "WalkMove": {"kwargs": {"s": walkMove_s}, "prob": walkMove_prob},
        }

        ########## [WALKER INITIALIZATION] IDENTIFY PARAMETER RANGES TO SAMPLE FROM ##########
        params_dict = {}  # {par: [value, min, max]} Sample from [min, max] for walker initialization
        for value, key in zip(values, keys):
            if "::" in key:  # production coefficient
                params_dict[key] = [value, 2500, 3000]
            else:  # SDME fit parameter
                params_dict[key] = [value, -1, 1]

        ############## RUN MCMC ##############
        print(f"Trial {trial.number}...")
        print(f"Moves mixture: {moves_mixture}")
        print(f"Parameter ranges: {params_dict}")

        # Seed the RNG for consistent walker initialization
        #   Do not want to introduce additional randomization in
        #   order to isolate the effect of the moves mixture
        np.random.seed(self.seed)

        with open(f"{self.ofolder}/mcmc_{trial.number}.yml", "w") as f:
            """ Create configuration for reproducibility """
            yaml.dump(
                {
                    "moves_mixture": moves_dict,
                    "params_dict": params_dict,
                    "ofolder": self.ofolder,
                    "ofile": ofile,
                    "nwalkers": self.nwalkers,
                    "burnIn": self.burnIn,
                    "nsamples": self.nsamples,
                    "seed": self.seed,
                    "cfgfile": self.cfgfile,
                    "corner_ofile": corner_ofile,
                },
                f,
            )

        mcmcMgr = mcmcManager(self.atis, self.LoadParametersSamplers, ofile)

        mcmcMgr.perform_mcmc(
            nwalkers=self.nwalkers,
            burnIn=self.burnIn,
            nsamples=self.nsamples,
            params_dict=params_dict,
            moves_mixture=moves_mixture,
        )

        mcmcMgr.draw_corner(corner_ofile)

        # HMM WHERE IS RESULTS DEFINED?
        return results["autocorr_time"], results["acceptance_fraction"]


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="emcee fitter")
    parser.add_argument("cfgfile", type=str, help="Config file name")
    parser.add_argument("--ntrials", type=int, default=10, help="Number of trials. Default 10")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for consistent walker random initialization. Default 42")
    parser.add_argument("--ofolder", type=str, default="mcmc", help='Output folder name. Default "mcmc"')
    parser.add_argument("--nwalkers", type=int, default=32, help="Number of walkers. Default 32")
    parser.add_argument("--burnin", type=int, default=100, help="Number of burn-in steps. Default 100")
    parser.add_argument("--nsamples", type=int, default=25000, help="Number of samples. Default 1000")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file if exists")
    parser.add_argument("--accelerator", type=str, default="", help='Force use of given "accelerator" ~ [gpu, mpi, mpigpu, gpumpi]. Default "" = cpu')

    args = parser.parse_args(sys.argv[1:])

    cfgfile = args.cfgfile
    ntrials = args.ntrials
    seed = args.seed
    ofolder = args.ofolder
    nwalkers = args.nwalkers
    burnIn = args.burnin
    nsamples = args.nsamples
    overwrite_ofile = args.overwrite

    print("\n ===================")
    print(f" cfgfile: {cfgfile}")
    print(f" ntrials: {ntrials}")
    print(f" seed: {seed}")
    print(f" ofolder: {ofolder}")
    print(f" nwalkers: {nwalkers}")
    print(f" burnIn: {burnIn}")
    print(f" nsamples: {nsamples}")
    print(f" overwrite_ofile: {overwrite_ofile}")
    print(" ===================\n")

    os.system(f"mkdir -p {ofolder}")

    ############## SET ENVIRONMENT VARIABLES ##############
    PYAMPTOOLS_HOME = os.environ["PYAMPTOOLS_HOME"]

    ################### LOAD LIBRARIES ##################
    atiSetup.setup(globals(), args.accelerator)

    ############## LOAD CONFIGURATION FILE ##############
    assert os.path.exists(cfgfile), "Config file does not exist at specified path"
    parser = ConfigFileParser(cfgfile)
    cfgInfo: ConfigurationInfo = parser.getConfigurationInfo()

    ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude(Zlm())
    AmpToolsInterface.registerAmplitude(BreitWigner())
    AmpToolsInterface.registerAmplitude(Piecewise())
    AmpToolsInterface.registerAmplitude(PhaseOffset())
    AmpToolsInterface.registerAmplitude(TwoPiAngles())
    AmpToolsInterface.registerDataReader(DataReader())
    AmpToolsInterface.registerDataReader(DataReaderFilter())

    ### CURRENTLY, ONLY ALLOWS A SINGLE CFGFILE ###
    atis = [AmpToolsInterface(cfgInfo)]
    LoadParametersSamplers = [LoadParameters(cfgInfo)]

    ############## RUN OPTIMIZATION ##############
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "mcmcOptimalMoves"
    storage_name = f"sqlite:///{ofolder}/mcmcOptimalMoves.db"
    study = optuna.create_study(
        directions=["minimize", "maximize"],  # multi-objective optimization
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,  # resume experiment if the database exists
    )

    conditions = {
        "atis": atis,
        "LoadParametersSamplers": LoadParametersSamplers,
        "cfgfile": cfgfile,  # for YAML output
        "cfgInfo": cfgInfo,
        "ofolder": ofolder,
        "nwalkers": nwalkers,
        "burnIn": burnIn,
        "nsamples": nsamples,
        "seed": seed,
        "overwrite_ofile": overwrite_ofile,
    }

    study.optimize(Objective(conditions), n_trials=ntrials)

    print("best trail parameter and objective value:")
    try:
        print(study.best_params)
        print(study.best_trial.value)
    except Exception:
        # If no best trial exists (i.e. all failed trials) then optuna will throw a
        #   RuntimeError crashing python which causes AmpTools to gracelessly terminate
        #   dumping a giant stacktrace which could mislead
        print("Catching raised error while retrieving the best trial, try and gracefully exit...")
