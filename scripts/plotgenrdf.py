#!/usr/bin/env python3

import ROOT
import os
from typing import List
import argparse
import numpy as np
from pyamptools.utility.plotgen_utils import book_histogram, turn_on_specifc_waveset, draw_histograms
from pyamptools import atiSetup


def _cli_plotgenrdf():

    ''' Command line interface for plotting fit results using RDataFrames '''

    ############## SET ENVIRONMENT VARIABLES ##############
    REPO_HOME     = os.environ['REPO_HOME']

    ################### LOAD LIBRARIES ##################
    atiSetup.setup(globals(), use_fsroot=True)

    from pyamptools.utility.RDFmacros import loadMacros
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
