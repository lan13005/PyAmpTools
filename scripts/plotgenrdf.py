#!/usr/bin/env python3

import ROOT
import os
from typing import List
import argparse
import numpy as np
from utils import remove_all_whitespace
from plotgen_utils import book_histogram, turn_on_specifc_waveset
import atiSetup

def draw_histograms(
    results,
    hist_output_name,
    particles,
    HISTS_TO_BOOK,
    amplitudes = 'all',
):
    '''
    Draw histograms from a FitResults object. Histograms are booked using the book_histogram() function that uses macros to compute
    kinematic quantities to plot. Booked histograms are lazily evaluated / filled with RDataFrame.

    Args:
        results (FitResults): FitResults object
        hist_output_name (str): Output file name, do not include file type
        particles (List[str]): List of particles in reaction
        HISTS_TO_BOOK (Dict[str, List]): Dictionary of histograms to book. See book_histogram() for details
        amplitudes (str): Space separated list of wavesets to turn on. Wavesets are semi-colon ; separated list of amplitudes. "all" turns on all waves.

    Returns:
        None, dumps a pdf file based on hist_output_name
    '''

    THStack = ROOT.THStack
    TCanvas = ROOT.TCanvas
    plotGen = ROOT.PlotGenerator( results )

    assert( '.' not in hist_output_name ), "Do not include file type in the output name ( -o flag )"

    N_BOOKED_HISTS = len(HISTS_TO_BOOK)
    N_PARTICLES = len(particles)

    kData, kBkgnd, kGenMC, kAccMC, kNumTypes = plotGen.kData, plotGen.kBkgnd, plotGen.kGenMC, plotGen.kAccMC, plotGen.kNumTypes
    kColors = {
        kData: ROOT.kBlack,
        kBkgnd: ROOT.kRed-9,
        kAccMC: ROOT.kGreen-8,
        kGenMC: ROOT.kAzure-4} # for the 4 data sources

    reactionNames =  list(results.reactionList())

    ### FOR EACH WAVESET, PLOT THE HISTOGRAMS ###
    amplitudes = amplitudes.split(' ')
    for amp in amplitudes:
        turn_on_specifc_waveset(plotGen, results, amp)

        HISTOGRAM_STORAGE = {} # {type: [hist1, hist2, ...]}
        DRAW_OPT_STORAGE = {}
        for srctype in [kData, kBkgnd, kGenMC, kAccMC]:

            ########### LOAD THE DATA ###########
            # Reaction: { Variable: [Values] } }
            value_map = plotGen.projected_values(reactionNames, srctype, N_PARTICLES)
            value_map = value_map[srctype]
            value_map = {k: np.array(v) for k,v in value_map}

            df = ROOT.RDF.FromNumpy(value_map)
            columns = df.GetColumnNames()

            ######### RESTRUCTURE DATA FOR NICER CALCULATIONS #########
            df.Define("GLUEXTARGET", "std::vector<float> p{0.938272, 0.0, 0.0, 0.0}; return p;")
            for i, particle in enumerate(particles):
                cmd = f"std::vector<float> p{{ PxP{i}, PyP{i}, PzP{i}, EnP{i} }}; return p;"
                df = df.Define(f"{particle}", cmd)
            # print(df.Describe())

            ######### BOOK HISTOGRAMS #########
            BOOKED_HISTOGRAMS, DRAW_OPTIONS = book_histogram(df, HISTS_TO_BOOK, columns)
            HISTOGRAM_STORAGE[srctype] = BOOKED_HISTOGRAMS
            DRAW_OPT_STORAGE[srctype] = DRAW_OPTIONS

        ###########################################################
        ### NOW CONFIGURE HOW YOU WANT TO DRAW THE HISTOGRAMS ####
        ### HERE IS AN EXAMPLE... BUT IMPOSSIBLE TO MAKE GENERIC #

        nrows = int(np.floor(np.sqrt(len(HISTS_TO_BOOK))))
        ncols = int(np.ceil(len(HISTS_TO_BOOK)/nrows))

        canvas = TCanvas("canvas", "canvas", 1440, 1080)
        canvas.Clear()
        canvas.Divide(ncols, nrows)

        output_name = hist_output_name + f"_{amp}"
        # canvas.Print(f"{output_name}.pdf[")
        stacks = []
        for ihist in range(N_BOOKED_HISTS):
            canvas.cd(ihist+1)
            data_hist = HISTOGRAM_STORAGE[kData][ihist]
            data_hist.SetMarkerStyle(ROOT.kFullCircle)
            data_hist.SetMinimum(0)
            data_hist.SetMarkerSize(1.0)
            data_hist.Draw('E') # Draw first to set labels and y-limits
            stacks.append(THStack("stack",""))
            for srctype in [kBkgnd, kAccMC]:
                booked_hist = HISTOGRAM_STORAGE[srctype][ihist]
                drawOptions = DRAW_OPT_STORAGE[srctype][ihist]
                booked_hist.SetFillColorAlpha(kColors[srctype],1.0)
                booked_hist.SetLineColor(0)
                booked_hist
                hist_ptr = booked_hist.GetPtr()
                stacks[-1].Add(hist_ptr)
            stacks[-1].Draw('HIST SAME')
            data_hist.Draw('E SAME') # Redraw data

        canvas.Print(f"{output_name}.png")
        # canvas.Print(f"{output_name}.pdf")
        # canvas.Print(f"{output_name}.pdf]")

        # THStack is drawn on TCanvas. Deleting TCanvas (which normally happens when it goes out of scope)
        #   before THStack will lead to improper deallocation. Also deleting elements of stacks in a for loop
        #   does not work. The entire object needs to be deleted
        del stacks
        del canvas

def _cli_plotgenrdf():

    ''' Command line interface for plotting fit results using RDataFrames '''

    ############## SET ENVIRONMENT VARIABLES ##############
    REPO_HOME     = os.environ['REPO_HOME']

    ################### LOAD LIBRARIES ##################
    atiSetup.setup(globals(), use_fsroot=True)

    from RDFmacros import loadMacros
    loadMacros()
    ### IF USING ROOT TO PLOT - CAN SET GLUEX STYLE HERE ###
    gInterpreter.ProcessLine('#include "gluex_style.h"')
    gluex_style = ROOT.gluex_style() # Returns TStyle
    gluex_style.SetPadRightMargin(0.08)
    gluex_style.cd()

    ############## ENABLE MULTI-THREADING ##############
    ROOT.ROOT.EnableImplicitMT() # REMOVE THIS WHEN DEBUGGING

    ############## PARSE COMMAND LINE ARGS ##############
    parser = argparse.ArgumentParser(description='PlotGenerator fit results with RDataFrames')
    parser.add_argument('fit_results', type=str, help='Path to fit results file')
    parser.add_argument('-o', '--output', type=str, default='plotgenrdf_result', help='Output file name, do not include file type')
    args = parser.parse_args()
    fit_results = args.fit_results
    hist_output_name = args.output

    results = FitResults( fit_results )
    if not results.valid():
        print(f'Invalid fit result in file: {fit_results}')
        exit()

    ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude( Zlm() )
    AmpToolsInterface.registerDataReader( DataReader() )

    ############### BOOKEEPING ################
    ## START WITH 1D HISTS SO WE CAN REUSE FUNCTION VALUES! ##
    ## FOR THIS EXAMPLE, WILL NOT INCLUDE 2D PLOTS SO WE CAN STACK 1D HISTS FOR FIT RESULTS ##
    HISTS_TO_BOOK = {
        # 1D Hists
        # HistName: [ xname, Function, title, n_bins, x-min, x-max, drawOptions]
        "Metapi": [ "Metapi", "MASS(ETA,PI0)", ";M(#eta#pi);Events", 50, 1.04, 1.72, "" ],
        "Meta":   [ "Meta", "MASS(ETA)", ";M(#eta);Events", 50, 0.49, 0.61, "" ],
        "Mpi0":   [ "Mpi0", "MASS(PI0)", ";M(#pi^{0});Events", 50, 0.1, 0.18, "" ],
        "cosGJ":  [ "cosGJ", "GJCOSTHETA(ETA,PI0,RECOIL)", ";cos(#theta_{GJ});Events", 50, -1, 1, "" ],
        "cosHel": [ "cosHel","HELCOSTHETA(ETA,PI0,RECOIL)", ";cos(#theta_{HEL});Events", 50, -1, 1, "" ],
        "phiHel": [ "phiHel","HELPHI(ETA,PI0,RECOIL,GLUEXBEAM)", ";#phi_{HEL};Events", 50, -1, 1, "" ],

        # 2D Hists
        # HistName:     [ xname, xfunction, title, nx_bins, x_min, x_max, yname, yfunction, ny_bins, y_min, y_max, drawOptions]
        # "cosHelvsMass": [ "Metapi", "MASS(ETA,PI0)", "M(#eta#pi) vs cos(#theta_{hel})", 100, 1.04, 1.72, "cosHel", "GJCOSTHETA(ETA,PI0,GLUEXBEAM)", 100, -1, 1, "COLZ"],
    }

    ############## SETUP ##############
    particles = ['GLUEXBEAM','RECOIL','ETA','PI0']

    ############## DRAW HISTOGRAMS ##############
    draw_histograms(results, hist_output_name, particles, HISTS_TO_BOOK)


if __name__ == '__main__':
    _cli_plotgenrdf()
