import argparse
from rich.console import Console

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("main_yaml", type=str)
    parser.add_argument("-c", "--clean", action="store_true", help="Do not use cached data, start clean")
    parser.add_argument("-npi", "--no_plot_intensities", action="store_true", help="Do not plot posterior distribution of amplitude intensities")
    parser.add_argument("-npc", "--no_plot_complex_plane", action="store_true", help="Do not plot posterior distribution in complex plane")
    parser.add_argument("-npo", "--no_plot_overview", action="store_true", help="Do not plot overview across bins (intensity and phases)")
    parser.add_argument("-npm", "--no_plot_moments", action="store_true", help="Do not plot moments across bins")
    parser.add_argument("-npr", "--no_plot_resonance_parameters", action="store_true", help="Do not plot resonance parameters")
    parser.add_argument("-nm",  "--no_montage", action="store_true", help="Do not montage all plots across bins")
    parser.add_argument("-log", "--log_scale", action="store_true", help="Plot intensities on log scale in overview plots (phases remain linear)")
    parser.add_argument("-norm", "--normalization_scheme", type=int, default=0, help="Normalization scheme for moments. 0: normalize to H0, 1: normalize to intenstiy, 2: keep raw moments")
    parser.add_argument("--min_mass", type=float, default=None, help="Minimum mass for plotting (GeV/c²). If None, use full range.")
    parser.add_argument("--max_mass", type=float, default=None, help="Maximum mass for plotting (GeV/c²). If None, use full range.")
    parser.add_argument("--ylim", type=float, default=None, help="Limit y-axis to this percentage of the would-be used axis limit (0.0-1.0). If None, use full range.")
    args = parser.parse_args()
    
    console = Console()
    
    # Put here to faster argparse
    from pyamptools.utility.resultManager import (
        ResultManager, 
        plot_binned_intensities, 
        plot_overview_across_bins, 
        plot_moments_across_bins, 
        montage_and_gif_select_plots,
        plot_binned_complex_plane,
        plot_gen_curves,
        plot_resonance_parameters
    )
        
    resultManager = ResultManager(args.main_yaml)
    if args.clean:
        resultManager.clean()
    resultManager.attempt_load_all()
    
    console.print(f"Will attempt to use moment normalization scheme: {args.normalization_scheme}")
    resultManager.attempt_project_moments(normalization_scheme=args.normalization_scheme)
    
    if not args.no_plot_intensities:
        plot_binned_intensities(resultManager, min_mass=args.min_mass, max_mass=args.max_mass)
    if not args.no_plot_complex_plane:
        plot_binned_complex_plane(resultManager, min_mass=args.min_mass, max_mass=args.max_mass)
    if not args.no_plot_overview:
        plot_overview_across_bins(resultManager, log_scale=args.log_scale, min_mass=args.min_mass, max_mass=args.max_mass, ylim=args.ylim)
    if not args.no_plot_moments:
        plot_moments_across_bins(resultManager, min_mass=args.min_mass, max_mass=args.max_mass)
    if not args.no_plot_resonance_parameters:
        plot_resonance_parameters(resultManager)
    if not args.no_montage:
        montage_and_gif_select_plots(resultManager)
    # Note: plot_gen_curves is not called by default in the original script, but adding it for completeness
    plot_gen_curves(resultManager, min_mass=args.min_mass, max_mass=args.max_mass)
    
    # without this, jax complains, probably some leak
    del resultManager