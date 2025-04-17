import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file", type=str)
    parser.add_argument("-npi", "--no_plot_intensities", action="store_true", help="Do not plot posterior distribution of amplitude intensities")
    parser.add_argument("-npc", "--no_plot_complex_plane", action="store_true", help="Do not plot posterior distribution in complex plane")
    parser.add_argument("-npo", "--no_plot_overview", action="store_true", help="Do not plot overview across bins (intensity and phases)")
    parser.add_argument("-npm", "--no_plot_moments", action="store_true", help="Do not plot moments across bins")
    parser.add_argument("-nm",  "--no_montage", action="store_true", help="Do not montage all plots across bins")
    args = parser.parse_args()
    
    # Put here to faster argparse
    from pyamptools.utility.resultManager import (
        ResultManager, 
        plot_binned_intensities, 
        plot_overview_across_bins, 
        plot_moments_across_bins, 
        montage_and_gif_select_plots,
        plot_binned_complex_plane
    )
        
    resultManager = ResultManager(args.yaml_file)
    resultManager.attempt_load_all()
    resultManager.attempt_project_moments()
    
    if not args.no_plot_intensities:
        plot_binned_intensities(resultManager)
    if not args.no_plot_complex_plane:
        plot_binned_complex_plane(resultManager)
    if not args.no_plot_overview:
        plot_overview_across_bins(resultManager)
    if not args.no_plot_moments:
        plot_moments_across_bins(resultManager)
    if not args.no_montage:
        montage_and_gif_select_plots(resultManager)