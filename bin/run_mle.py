import os
os.environ["JAX_PLATFORMS"] = "cpu"
from pyamptools.utility.general import load_yaml, Timer
import numpy as np
import sys
import pickle as pkl
from iminuit import Minuit
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from pyamptools.utility.opt_utils import Objective
from rich.console import Console
import logging

Minuit.errordef = Minuit.LIKELIHOOD
console = Console()

# TODO:
# - NLOPT - another set of optimizers to consider. Could have better global optimization algorithms

# NOTE:
# - Regularization is currently handled by the pwa manager class which will reference the key in the yaml file
#       regularization:
#           apply_regularization: false
#           method: none
#           lambdas: 0.0
#           en_alpha: 0.0
# - lambdas can be a dictionary of reg. strengths, i.e. {"Sp0+": 1e2, "Dm1-": 1e1, "Dp0-": 1e0}

def run_single_bin_fits(
    job_info
    ):
    """
    Run a set of randomly initialized fits in a single bin
    
    Args: job_info contains the following:
        pwa_manager: GluexJaxManager instance
        bin_idx: Bin index to fit
        bin_seed: Random seed for this bin for reproducibility
        n_iterations: Number of random initializations to perform
        scale: Scale for random initialization, real/imag parts are sampled from uniform [-scale, scale]
        method: Optimization method to use
    """
    pwa_manager, bin_idx, bin_seed, n_iterations, scale, method = job_info
    
    # Create a bin-specific console if writing to files
    bin_console = console
    if not dump_to_stdout:
        log_file = f"{output_folder}/{method}_bin{bin_idx}.log"
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        bin_console = Console(file=open(log_file, "w"), highlight=False)
    
    pwa_manager.set_bins(np.array([bin_idx]))
    np.random.seed(bin_seed)
    seed_list = np.random.randint(0, 1000000, n_iterations)
    
    def run_single_random_fit(initial_guess):
        """Single fit in a single bin"""
                
        reference_waves = main_dict["phase_reference"].split("_")
        acceptance_correct = main_dict["acceptance_correct"]
        obj = Objective(pwa_manager, bin_idx, nPars, nmbMasses, nmbTprimes, reference_waves=reference_waves)
        
        initial_likelihood = obj.objective(initial_guess).item()
        bin_console.print("\n**************************************************************", style="bold")
        bin_console.print(f"Bin {bin_idx} | Iteration {iteration_idx}", style="bold")
        bin_console.print(f"Initial likelihood: {initial_likelihood}", style="bold")
        bin_console.print(f"Using method: {method}", style="bold")
        
        final_result_dict = {}
        optim_result = {}
        if method == 'minuit-analytic':
            from pyamptools.utility.opt_utils import optimize_single_bin_minuit
            optim_result = optimize_single_bin_minuit(obj, initial_guess, bin_idx, use_analytic_grad=True)
        elif method == 'minuit-numeric':
            from pyamptools.utility.opt_utils import optimize_single_bin_minuit
            optim_result = optimize_single_bin_minuit(obj, initial_guess, bin_idx, use_analytic_grad=False)
        elif method == 'L-BFGS-B' or method == 'trust-ncg' or method == 'trust-krylov':
            from pyamptools.utility.opt_utils import optimize_single_bin_scipy
            optim_result = optimize_single_bin_scipy(obj, initial_guess, bin_idx, method=method)
        else:
            raise ValueError(f"Invalid Maximum Likelihood Based method: {method}")
        
        # NOTE: Here we force the usage of Tikhonov regularized covariance matrix to obtain the intensity and intensity errors
        #       Minuit does not always return a covariance matrix (i.e. if it fails). Tikhonov regularization adds a small value
        #       to the diagonal of the Hessian to ensure it is positive definite before inversion
        
        final_params = np.array(optim_result['parameters'])
        intensity, intensity_error = obj.intensity_and_error(final_params, optim_result['covariance']['tikhonov'], pwa_manager.waveNames, acceptance_correct=acceptance_correct)
        final_result_dict['intensity'] = intensity
        final_result_dict['intensity_err'] = intensity_error
        for wave in pwa_manager.waveNames:
            intensity, intensity_error = obj.intensity_and_error(final_params, optim_result['covariance']['tikhonov'], [wave], acceptance_correct=acceptance_correct)
            final_result_dict[wave] = intensity
            final_result_dict[f"{wave}_err"] = intensity_error
        final_result_dict['likelihood'] = optim_result['likelihood']
        final_result_dict['initial_likelihood'] = initial_likelihood
        
        _intensity = final_result_dict['intensity'] if final_result_dict['intensity'] is not None else np.nan
        _intensity_err = final_result_dict['intensity_err'] if final_result_dict['intensity_err'] is not None else np.nan
        bin_console.print(f"\nTotal Intensity: value = {_intensity:<10.5f} +- {_intensity_err:<10.5f}", style="bold")
        for iw, wave in enumerate(pwa_manager.waveNames):
            real_part = final_params[2*iw]
            imag_part = final_params[2*iw+1]
            real_errs = {}
            imag_errs = {}
            for key in optim_result['covariance'].keys():
                real_err = optim_result['covariance'][key]
                imag_err = optim_result['covariance'][key]
                real_errs[key] = 0 # Default to 0 if none or non-positive
                imag_errs[key] = 0 # Default to 0 if none or non-positive
                if real_err is not None and real_err[2*iw, 2*iw] > 0:
                    real_errs[key] = np.sqrt(real_err[2*iw, 2*iw])
                if imag_err is not None and imag_err[2*iw+1, 2*iw+1] > 0:
                    imag_errs[key] = np.sqrt(imag_err[2*iw+1, 2*iw+1])
            methods = list(real_errs.keys())
            real_part_errs = [f"{real_errs[m]:<10.5f}" for m in methods]
            imag_part_errs = [f"{imag_errs[m]:<10.5f}" for m in methods]
            bounds = {}
            bounds['rlb'] = optim_result['bounds'][2*iw][0]     # real part lower bound
            bounds['rub'] = optim_result['bounds'][2*iw][1]     # real part upper bound
            bounds['ilb'] = optim_result['bounds'][2*iw+1][0]   # imag part lower bound
            bounds['iub'] = optim_result['bounds'][2*iw+1][1]   # imag part upper bound
            for key in bounds.keys():
                if bounds[key] is None: bounds[key] = "None"
            _intensity = final_result_dict[wave] if final_result_dict[wave] is not None else np.nan
            _intensity_err = final_result_dict[f'{wave}_err'] if final_result_dict[f'{wave}_err'] is not None else np.nan
            bin_console.print(f"{wave:<10}  Intensity: value = {_intensity:<10.5f} +- {_intensity_err:<10.5f}", style="bold")
            bin_console.print(f"            Real part: value = {real_part:<10.5f} | Errors ({', '.join(methods)}) = ({', '.join(real_part_errs)}) | Bounds: \[{bounds['rlb']:<10}, {bounds['rub']:<10}]) ", style="bold")
            bin_console.print(f"            Imag part: value = {imag_part:<10.5f} | Errors ({', '.join(methods)}) = ({', '.join(imag_part_errs)}) | Bounds: \[{bounds['ilb']:<10}, {bounds['iub']:<10}]) ", style="bold")

        bin_console.print(f"Optimization results:", style="bold")
        bin_console.print(f"Success: {optim_result['success']}", style="bold")
        bin_console.print(f"Final likelihood: {optim_result['likelihood']}", style="bold")
        bin_console.print(f"Percent negative eigenvalues: {optim_result['hessian_diagnostics']['fraction_negative_eigenvalues']:0.3f}", style="bold")
        bin_console.print(f"Percent small eigenvalues: {optim_result['hessian_diagnostics']['fraction_small_eigenvalues']:0.3f}", style="bold")
        bin_console.print(f"Message: {optim_result['message']}", style="bold")
        bin_console.print("**************************************************************\n", style="bold")
        
        final_par_values = {}
        for iw, wave in enumerate(pwa_manager.waveNames):
            final_par_values[f"{wave}_amp"] = complex(final_params[2*iw], final_params[2*iw+1]) # type convert away from np.complex to python complex
        
        final_result_dict['initial_guess_dict'] = initial_guess_dict
        final_result_dict['final_par_values'] = final_par_values
        final_result_dict['covariances'] = optim_result['covariance']

        return final_result_dict
        ### End of run_single_random_fit function ###
    
    final_result_dicts = []
    for iteration_idx in range(n_iterations):
        np.random.seed(seed_list[iteration_idx])
        initial_guess = scale * np.random.uniform(-1, 1, nPars)
        initial_guess_dict = {} 
        for iw, wave in enumerate(waveNames):
            if wave in reference_waves:
                # Rotate away the imaginary part of the reference wave as to not bias initialization small for reference waves
                initial_guess[2*iw] = np.abs(initial_guess[2*iw] + 1j * initial_guess[2*iw+1])
                initial_guess[2*iw+1] = 0
                initial_guess_dict[f"{wave}_amp"] = complex(initial_guess[2*iw], 0)
            else:
                initial_guess_dict[f"{wave}_amp"] = complex(initial_guess[2*iw], initial_guess[2*iw+1])
        final_result_dict = run_single_random_fit(initial_guess)
        final_result_dicts.append(final_result_dict)

    ofile = f"{output_folder}/{method}_bin{bin_idx}.pkl"
    with open(ofile, "wb") as f:
        pkl.dump(final_result_dicts, f)
    bin_console.print(f"Saved results to {ofile}", style="bold")
    
    # Close the file if we opened one
    if not dump_to_stdout:
        bin_console.file.close()

class OptimizerHelpFormatter(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"Error: {message}\n")
        self.print_help()
        sys.exit(2)
    
    def print_help(self, file=None):
        help_text = self.format_help()
        console.print(help_text)

    def format_help(self):
        help_message = super().format_help()
        
        method_help = "\n[bold green]Optimizer Descriptions:[/bold green]\n"
        method_help += "\n[bold blue]Minuit-based Methods:[/bold blue]\n"
        method_help += "  * minuit-numeric:\n"
        method_help += "      Let Minuit compute numerical gradients\n"
        method_help += "  * minuit-analytic:\n"
        method_help += "      Uses analytic gradients from PWA likelihood manager\n"
        
        method_help += "\n[bold blue]SciPy-based Methods:[/bold blue]\n"
        method_help += "  * L-BFGS-B:\n"
        method_help += "      Limited-memory BFGS quasi-Newton method (stores approximate Hessian)\n"
        method_help += "      + Efficient for large-scale problems\n"
        method_help += "      - May struggle with highly correlated parameters\n"
        
        method_help += "  * trust-ncg [bold yellow](limited testing)[/bold yellow]:\n"
        method_help += "      Trust-region Newton-Conjugate Gradient\n"
        method_help += "      + Adaptively adjusts step size using local quadratic approximation\n"
        method_help += "      + Efficient for large-scale problems\n"
        method_help += "      - Can be unstable for ill-conditioned problems\n"
        
        method_help += "  * trust-krylov [bold yellow](limited testing)[/bold yellow]:\n"
        method_help += "      Trust-region method with Krylov subspace solver\n"
        method_help += "      + Better handling of indefinite (sparse) Hessians, Kyrlov subspcae accounts for non-Euclidean geometry\n"
        method_help += "      + More robust for highly correlated parameters\n"
        
        return help_message + "\n" + method_help

if __name__ == "__main__":
    parser = OptimizerHelpFormatter(description="Run optimization fits using various methods.")
    parser.add_argument("main_yaml", type=str,
                       help="Path to PyAmpTools YAML configuration file")
    parser.add_argument("-b", "--bins", type=int, nargs="+",
                       help=f"List of bin indicies to process (default: all bins)")
    parser.add_argument("-np", "--n_processes", type=int, default=None,
                        help="Number of processes to run in parallel")
    
    ##### OPTIMIZATION METHOD ARGS #####
    # NOTE: L-BFGS-B is found to be very fast but Minuit appears to be more robust so set as default
    #       COMPASS uses LBFGS for their binned fits but minuit for their mass dependent fits due to L-BFGS-B performing worse
    parser.add_argument("-m", "--method", type=str, 
                       choices=['minuit-numeric', 'minuit-analytic', 'L-BFGS-B'], # , 'trust-ncg', 'trust-krylov'], 
                       default=None,
                       help="Optimization method to use")
    
    ##### RANDOM INITIALIZATION ARGS #####
    parser.add_argument("-n", "--n_random_intializations", type=int, default=None,
                        help="Number of random initializations to perform")
    parser.add_argument("-sc", "--scale", type=float, default=None,
                        help="Uniformly sample starting real/imag parts of the amplitude on [-scale, scale]")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Seed for main random number generator")
    
    #### HELPFUL ARGS ####
    parser.add_argument("--print_wave_names", action="store_true",
                       help="Print wave names without running any fits")
    parser.add_argument("--stdout", action="store_true",
                       help="Print output to stdout instead of dumping to a log file")
    
    args = parser.parse_args()
    
    #### LOAD YAML FILES ####
    main_dict = load_yaml(args.main_yaml)
    iftpwa_dict = main_dict["nifty"]["yaml"] # DictConfig ~ Dict-like object
    if not iftpwa_dict:
        raise ValueError("iftpwa YAML file is required")
    if not main_dict:
        raise ValueError("Main YAML file is required")
    waveNames = main_dict["waveset"].split("_")
    nmbMasses = main_dict["n_mass_bins"]
    nmbTprimes = main_dict["n_t_bins"]
    nPars = 2 * len(waveNames)
    reference_waves = main_dict["phase_reference"].split("_")
    
    #### EXIT IF ONLY PRINTING WAVE NAMES
    if args.print_wave_names:
        console.print(f"Wave names: {waveNames}", style="bold")
        sys.exit(0)
    
    #### PARSE ARGS ####
    from pyamptools.utility.general import fyea
    seed = fyea(main_dict["mle"], "seed", args.seed)
    scale = fyea(main_dict["mle"], "scale", args.scale)
    n_iterations = fyea(main_dict["mle"], "n_random_intializations", args.n_random_intializations)
    n_processes = fyea(main_dict, "n_processes", args.n_processes)
    method = fyea(main_dict["mle"], "method", args.method)
    bins = fyea(main_dict["mle"], "bins", args.bins)
    dump_to_stdout = args.stdout
    
    np.random.seed(seed)

    ##### LOAD DEFAULTS IF NONE PROVIDED #####
    bins = fyea(main_dict["mle"], "bins", args.bins)
    if bins is None:
        bins = list(np.arange(nmbMasses * nmbTprimes))
    if len(bins) != len(set(bins)):
        console.print("Warning: user requested repeated bins, ignoring repeated bins", style="bold yellow")
        bins = sorted(list(set(bins)))  
    
    output_folder = os.path.join(main_dict["base_directory"], "MLE")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Check for any bins that are already processed
    for bin_idx in bins.copy():
        check_path = os.path.join(output_folder, f"{method}_bin{bin_idx}.pkl")
        if os.path.exists(check_path):
            bins.remove(bin_idx)
    if len(bins) == 0:
        console.print("No bins to process, exiting", style="bold red")
        sys.exit(0)
    console.print(f"Found {nmbMasses*nmbTprimes - len(bins)}/{nmbMasses*nmbTprimes} bins already processed", style="bold green")
    console.print(f"  Need to process {len(bins)}/{nmbMasses*nmbTprimes} bins", style="bold yellow")

    ##### LOAD PWA MANAGER #####
    from iftpwa1.pwa.gluex.gluex_jax_manager import (
        GluexJaxManager,
    )
    pwa_manager = GluexJaxManager(comm0=None, mpi_offset=1,
                                yaml_file=main_dict,
                                resolved_secondary=iftpwa_dict, prior_simulation=False, sum_returned_nlls=False, 
                                logging_level=logging.WARNING)

    ##### CREATE JOB ASSIGNMENTS #####
    job_assignments = {}
    job_counter = 0
    bin_seeds = np.random.randint(0, 100000000, len(bins))
    for bin_idx, seed in zip(bins, bin_seeds):
        job_assignments[job_counter] = (bin_idx, seed)
        job_counter += 1
    total_jobs = len(job_assignments)
    console.print(f"Total jobs: {total_jobs} -- distributing across {n_processes} processes", style="bold yellow")
    
    # copy input yaml_file to output_folder for reproducibility
    os.system(f"cp {args.main_yaml} {output_folder}/{os.path.basename(args.main_yaml)}")
    console.print(f"Copied {args.main_yaml} to {output_folder}/{os.path.basename(args.main_yaml)}", style="bold blue")
        
    ##### RUN JOBS IN PARALLEL #####
    timer = Timer()
    with Pool(processes=n_processes) as pool:
        job_args = [(pwa_manager, bin_idx, bin_seed, n_iterations, scale, method) for job_idx, (bin_idx, bin_seed) in job_assignments.items()]
        # Distribute work across processes
        with tqdm(total=total_jobs, desc="Processing jobs", unit="job") as pbar:
            for _ in pool.imap_unordered(run_single_bin_fits, job_args):
                pbar.update(1)
    console.print(f"Total time elapsed: {timer.read()[2]}", style="bold")

    sys.exit(0)

