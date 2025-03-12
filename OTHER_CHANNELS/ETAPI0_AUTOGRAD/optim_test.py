from pyamptools.utility.general import load_yaml, Timer
import numpy as np
import sys
import pickle as pkl
from iminuit import Minuit
import argparse
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
import os

from optimize_utility import Objective

from rich.console import Console

from iftpwa1.pwa.gluex.constants import INTENSITY_AC

Minuit.errordef = Minuit.LIKELIHOOD
comm0 = None
rank = 0 
mpi_offset = 1

console = Console()

# TODO:
# - NLOPT - another set of optimizers to consider. Could have better global optimization algorithms

def run_fit(
    pyamptools_yaml, 
    iftpwa_yaml,
    bin_idx, 
    iteration_idx,
    initial_guess, 
    initial_guess_dict,
    method="L-BFGS-B",
    ):
    """
    Run fit with specified method
    
    Args:
        bin_idx: Index of bin to optimize
        iteration_idx: Iteration number
        initial_guess: Initial parameters
        initial_guess_dict: Dictionary of initial parameters
        method: see argument parser for more information
        
        initial_guess contains full sized array across kinematic bins needed for GlueXJaxManager
        initial_guess_dict contains initial guess for this bin only stored as a comparison in the pkl file
    """
    
    from iftpwa1.pwa.gluex.gluex_jax_manager import (
        GluexJaxManager,
    )

    pwa_manager = GluexJaxManager(comm0=comm0, mpi_offset=mpi_offset,
                                yaml_file=pyamptools_yaml,
                                resolved_secondary=iftpwa_yaml, prior_simulation=False, sum_returned_nlls=False)
 
    pwa_manager.set_bins(np.array([bin_idx]))
    
    is_mle_method = method in ['minuit-numeric', 'minuit-analytic', 'L-BFGS-B', 'trust-ncg', 'trust-krylov']
    
    reference_waves = pyamptools_yaml["phase_reference"].split("_")
    obj = Objective(pwa_manager, bin_idx, nPars, nmbMasses, nmbTprimes, reference_waves=reference_waves)
    
    initial_likelihood = obj.objective(initial_guess).item()
    console.print("\n**************************************************************", style="bold")
    console.print(f"Bin {bin_idx} | Iteration {iteration_idx}", style="bold")
    console.print(f"Initial likelihood: {initial_likelihood}", style="bold")
    console.print(f"Using method: {method}", style="bold")
    
    final_result_dict = {}
    optim_result = {}
    if is_mle_method:
        if method == 'minuit-analytic':
            from optimize_utility import optimize_single_bin_minuit
            optim_result = optimize_single_bin_minuit(obj, initial_guess, bin_idx, use_analytic_grad=True)
        elif method == 'minuit-numeric':
            from optimize_utility import optimize_single_bin_minuit
            optim_result = optimize_single_bin_minuit(obj, initial_guess, bin_idx, use_analytic_grad=False)
        elif method == 'L-BFGS-B' or method == 'trust-ncg' or method == 'trust-krylov':
            from optimize_utility import optimize_single_bin_scipy
            optim_result = optimize_single_bin_scipy(obj, initial_guess, bin_idx, method=method)
        else:
            raise ValueError(f"Invalid Maximum Likelihood Based method: {method}")
        
        final_params = np.array(optim_result['parameters'])
        final_result_dict['total'] = obj.intensity(final_params)
        for wave in pwa_manager.waveNames:
            final_result_dict[wave] = obj.intensity(final_params, suffix=[wave])
        final_result_dict['likelihood'] = optim_result['likelihood']
        final_result_dict['initial_likelihood'] = initial_likelihood

    for iw, wave in enumerate(pwa_manager.waveNames):
        real_part = final_params[2*iw]
        imag_part = final_params[2*iw+1]
        real_errs = {}
        imag_errs = {}
        for key in optim_result['covariance'].keys():
            real_err = optim_result['covariance'][key]
            imag_err = optim_result['covariance'][key]
            if real_err is not None:
                real_errs[key] = np.sqrt(real_err[2*iw, 2*iw])
            if imag_err is not None:
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
        console.print(f"{wave:<10}  Intensity: value = {final_result_dict[wave]:<10.5f}", style="bold")
        console.print(f"            Real part: value = {real_part:<10.5f} | Errors ({', '.join(methods)}) = ({', '.join(real_part_errs)}) | Bounds: \[{bounds['rlb']:<10}, {bounds['rub']:<10}]) ", style="bold")
        console.print(f"            Imag part: value = {imag_part:<10.5f} | Errors ({', '.join(methods)}) = ({', '.join(imag_part_errs)}) | Bounds: \[{bounds['ilb']:<10}, {bounds['iub']:<10}]) ", style="bold")

    console.print(f"Optimization results:", style="bold")
    console.print(f"Success: {optim_result['success']}", style="bold")
    console.print(f"Final likelihood: {optim_result['likelihood']}", style="bold")
    console.print(f"Message: {optim_result['message']}", style="bold")
    console.print("**************************************************************\n", style="bold")
    
    final_par_values = {}
    for iw, wave in enumerate(pwa_manager.waveNames):
        final_par_values[wave] = complex(final_params[2*iw], final_params[2*iw+1]) # type convert away from np.complex to python complex
    
    final_result_dict['initial_guess_dict'] = initial_guess_dict
    final_result_dict['final_par_values'] = final_par_values
    final_result_dict['covariances'] = optim_result['covariance']

    return final_result_dict

class OptimizerHelpFormatter(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"Error: {message}\n")
        self.print_help()
        sys.exit(2)

    def format_help(self):
        help_message = super().format_help()
        
        method_help = "\nOptimizer Descriptions:\n"
        method_help += "\nMinuit-based Methods:\n"
        method_help += "  * minuit-numeric:\n"
        method_help += "      Lets Minuit compute numerical gradients\n"
        method_help += "  * minuit-analytic:\n"
        method_help += "      Uses analytic gradients from PWA likelihood manager\n"
        
        method_help += "\nSciPy-based Methods:\n"
        method_help += "  * L-BFGS-B:\n"
        method_help += "      Limited-memory BFGS quasi-Newton method (stores approximate Hessian)\n"
        method_help += "      + Efficient for large-scale problems\n"
        method_help += "      - May struggle with highly correlated parameters\n"
        
        method_help += "  * trust-ncg:\n"
        method_help += "      Trust-region Newton-Conjugate Gradient\n"
        method_help += "      + Adaptively adjusts step size using local quadratic approximation\n"
        method_help += "      + Efficient for large-scale problems\n"
        method_help += "      - Can be unstable for ill-conditioned problems\n"
        
        method_help += "  * trust-krylov:\n"
        method_help += "      Trust-region method with Krylov subspace solver\n"
        method_help += "      + Better handling of indefinite (sparse) Hessians, Kyrlov subspcae accounts for non-Euclidean geometry\n"
        method_help += "      + More robust for highly correlated parameters\n"
        
        return help_message + "\n" + method_help

if __name__ == "__main__":
    parser = OptimizerHelpFormatter(description="Run optimization fits using various methods.")
    parser.add_argument("yaml_file", type=str,
                       help="Path to PyAmpTools YAML configuration file")    
    parser.add_argument("--method", type=str, 
                       choices=['minuit-numeric', 'minuit-analytic', 'L-BFGS-B', 'trust-ncg', 'trust-krylov'], 
                       default='L-BFGS-B',
                       help="Optimization method to use")
    parser.add_argument("--bins", type=int, nargs="+",
                       help="List of bin indices to process")
    parser.add_argument("--setting", type=int, default=0)
    
    #### HELPFUL ARGS ####
    parser.add_argument("--print_wave_names", action="store_true",
                       help="Print wave names")
    
    
    args = parser.parse_args()

    np.random.seed(42)
    scale = 50 # scale of uniform random initial guess
    n_iterations = 20 # 20 # number of randomized initial parameter fits to perform
    
    pyamptools_yaml = load_yaml(args.yaml_file)
    iftpwa_yaml = pyamptools_yaml["nifty"]["yaml"]
    iftpwa_yaml = load_yaml(iftpwa_yaml)
    
     # Lasso
     # - need lambda of 10 for lasso to do anything
     
     # Ridge
     # - lambda=10 on D-waves is too high
     # - lambda=1 on D-waves minimally alters D-waves but pushes up S-wave

    # Elastic Net
    # - alpha=0.5 seems to lean too heavily towards one side (ridge?) results look similar
    
    # settings = [
    #     {"lambdas": 1e2, "en_alpha": 1.0},
    #     {"lambdas": 1e1, "en_alpha": 1.0},
    #     {"lambdas": 1e0, "en_alpha": 1.0},
    #     {"lambdas": 1e-1, "en_alpha": 1.0},
    #     {"lambdas": 1e-2, "en_alpha": 1.0},
    #     {"lambdas": 1e2, "en_alpha": 0.0},
    #     {"lambdas": 1e1, "en_alpha": 0.0},
    #     {"lambdas": 1e0, "en_alpha": 0.0},
    #     {"lambdas": 1e-1, "en_alpha": 0.0},
    #     {"lambdas": 1e-2, "en_alpha": 0.0},
    #     {"lambdas": 1e2, "en_alpha": 0.5},
    #     {"lambdas": 1e1, "en_alpha": 0.5},
    #     {"lambdas": 1e0, "en_alpha": 0.5},
    #     {"lambdas": 1e-1, "en_alpha": 0.5},
    #     {"lambdas": 1e-2, "en_alpha": 0.5},
    #     {"lambdas": {"Dm2-": 1e2, "Dm1-": 1e2, "Dp0-": 1e2, "Dp1-": 1e2, "Dp2-": 1e2, "Dm2+": 1e2, "Dm1+": 1e2, "Dp0+": 1e2, "Dp1+": 1e2, "Dp2+": 1e2}, "en_alpha": 1.0},
    #     {"lambdas": {"Dm2-": 1e1, "Dm1-": 1e1, "Dp0-": 1e1, "Dp1-": 1e1, "Dp2-": 1e1, "Dm2+": 1e1, "Dm1+": 1e1, "Dp0+": 1e1, "Dp1+": 1e1, "Dp2+": 1e1}, "en_alpha": 1.0},
    #     {"lambdas": {"Dm2-": 1e0, "Dm1-": 1e0, "Dp0-": 1e0, "Dp1-": 1e0, "Dp2-": 1e0, "Dm2+": 1e0, "Dm1+": 1e0, "Dp0+": 1e0, "Dp1+": 1e0, "Dp2+": 1e0}, "en_alpha": 1.0},
    #     {"lambdas": {"Dm2-": 1e-1, "Dm1-": 1e-1, "Dp0-": 1e-1, "Dp1-": 1e-1, "Dp2-": 1e-1, "Dm2+": 1e-1, "Dm1+": 1e-1, "Dp0+": 1e-1, "Dp1+": 1e-1, "Dp2+": 1e-1}, "en_alpha": 1.0},
    #     {"lambdas": {"Dm2-": 1e-2, "Dm1-": 1e-2, "Dp0-": 1e-2, "Dp1-": 1e-2, "Dp2-": 1e-2, "Dm2+": 1e-2, "Dm1+": 1e-2, "Dp0+": 1e-2, "Dp1+": 1e-2, "Dp2+": 1e-2}, "en_alpha": 1.0},
    #     {"lambdas": {"Dm2-": 1e2, "Dm1-": 1e2, "Dp0-": 1e2, "Dp1-": 1e2, "Dp2-": 1e2, "Dm2+": 1e2, "Dm1+": 1e2, "Dp0+": 1e2, "Dp1+": 1e2, "Dp2+": 1e2}, "en_alpha": 0.0},
    #     {"lambdas": {"Dm2-": 1e1, "Dm1-": 1e1, "Dp0-": 1e1, "Dp1-": 1e1, "Dp2-": 1e1, "Dm2+": 1e1, "Dm1+": 1e1, "Dp0+": 1e1, "Dp1+": 1e1, "Dp2+": 1e1}, "en_alpha": 0.0},
    #     {"lambdas": {"Dm2-": 1e0, "Dm1-": 1e0, "Dp0-": 1e0, "Dp1-": 1e0, "Dp2-": 1e0, "Dm2+": 1e0, "Dm1+": 1e0, "Dp0+": 1e0, "Dp1+": 1e0, "Dp2+": 1e0}, "en_alpha": 0.0},
    #     {"lambdas": {"Dm2-": 1e-1, "Dm1-": 1e-1, "Dp0-": 1e-1, "Dp1-": 1e-1, "Dp2-": 1e-1, "Dm2+": 1e-1, "Dm1+": 1e-1, "Dp0+": 1e-1, "Dp1+": 1e-1, "Dp2+": 1e-1}, "en_alpha": 0.0},
    #     {"lambdas": {"Dm2-": 1e-2, "Dm1-": 1e-2, "Dp0-": 1e-2, "Dp1-": 1e-2, "Dp2-": 1e-2, "Dm2+": 1e-2, "Dm1+": 1e-2, "Dp0+": 1e-2, "Dp1+": 1e-2, "Dp2+": 1e-2}, "en_alpha": 0.0},
    #     {"lambdas": {"Dm2-": 1e2, "Dm1-": 1e2, "Dp0-": 1e2, "Dp1-": 1e2, "Dp2-": 1e2, "Dm2+": 1e2, "Dm1+": 1e2, "Dp0+": 1e2, "Dp1+": 1e2, "Dp2+": 1e2}, "en_alpha": 0.5},
    #     {"lambdas": {"Dm2-": 1e1, "Dm1-": 1e1, "Dp0-": 1e1, "Dp1-": 1e1, "Dp2-": 1e1, "Dm2+": 1e1, "Dm1+": 1e1, "Dp0+": 1e1, "Dp1+": 1e1, "Dp2+": 1e1}, "en_alpha": 0.5},
    #     {"lambdas": {"Dm2-": 1e0, "Dm1-": 1e0, "Dp0-": 1e0, "Dp1-": 1e0, "Dp2-": 1e0, "Dm2+": 1e0, "Dm1+": 1e0, "Dp0+": 1e0, "Dp1+": 1e0, "Dp2+": 1e0}, "en_alpha": 0.5},
    #     {"lambdas": {"Dm2-": 1e-1, "Dm1-": 1e-1, "Dp0-": 1e-1, "Dp1-": 1e-1, "Dp2-": 1e-1, "Dm2+": 1e-1, "Dm1+": 1e-1, "Dp0+": 1e-1, "Dp1+": 1e-1, "Dp2+": 1e-1}, "en_alpha": 0.5},
    #     {"lambdas": {"Dm2-": 1e-2, "Dm1-": 1e-2, "Dp0-": 1e-2, "Dp1-": 1e-2, "Dp2-": 1e-2, "Dm2+": 1e-2, "Dm1+": 1e-2, "Dp0+": 1e-2, "Dp1+": 1e-2, "Dp2+": 1e-2}, "en_alpha": 0.5},
    # ]
    
    # for k, v in settings[args.setting].items():
    #     pyamptools_yaml["regularization"][k] = v
    
    if not iftpwa_yaml:
        raise ValueError("iftpwa YAML file is required")
    if not pyamptools_yaml:
        raise ValueError("PyAmpTools YAML file is required")
    
    waveNames = pyamptools_yaml["waveset"].split("_")
    nmbMasses = pyamptools_yaml["n_mass_bins"]
    nmbTprimes = pyamptools_yaml["n_t_bins"]
    nPars = 2 * len(waveNames)
    reference_waves = pyamptools_yaml["phase_reference"].split("_")
    
    if args.print_wave_names:
        console.print(f"Wave names: {waveNames}", style="bold")
        sys.exit(0)

    seed_list = np.random.randint(0, 1000000, n_iterations)

    timer = Timer()
    if args.bins is None:
        raise ValueError("list of bin indices is required")

    final_result_dicts = []
    for bin_idx in args.bins:
        for iteration_idx in range(n_iterations):
            np.random.seed(seed_list[iteration_idx])
            initial_guess = scale * np.random.randn(nPars)
            initial_guess_dict = {} 
            for iw, wave in enumerate(waveNames):
                if wave in reference_waves:
                    # Rotate away the imaginary part of the reference wave as to not bias initialization small for reference waves
                    initial_guess[2*iw] = np.abs(initial_guess[2*iw] + 1j * initial_guess[2*iw+1])
                    initial_guess[2*iw+1] = 0
                    initial_guess_dict[wave] = complex(initial_guess[2*iw], 0)
                else:
                    initial_guess_dict[wave] = complex(initial_guess[2*iw], initial_guess[2*iw+1])
            final_result_dict = run_fit(
                pyamptools_yaml, 
                iftpwa_yaml, 
                bin_idx, 
                iteration_idx,
                initial_guess, 
                initial_guess_dict,
                method=args.method,
            )
            final_result_dicts.append(final_result_dict)
    
    sbins = "_".join([str(b) for b in args.bins]) if isinstance(args.bins, list) else args.bins
    if not os.path.exists("COMPARISONS"):
        os.makedirs("COMPARISONS")
    with open(f"COMPARISONS/{args.method}_bin{sbins}_setting{args.setting}.pkl", "wb") as f:
        pkl.dump(final_result_dicts, f)

    console.print(f"Total time elapsed: {timer.read()[2]}", style="bold")

    sys.exit(0)
