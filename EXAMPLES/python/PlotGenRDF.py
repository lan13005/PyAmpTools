#!/usr/bin/env python3

import ROOT
import os
from typing import List
import argparse
import numpy as np
from utils import remove_all_whitespace
from plotgen_utils import book_histogram


############## SET ENVIRONMENT VARIABLES ##############
REPO_HOME     = os.environ['REPO_HOME']
os.environ['ATI_USE_MPI'] = "0" # set to 1 to use MPI libraries
os.environ['ATI_USE_GPU'] = "0"
from atiSetup import *
from RDFmacros import loadMacros
loadMacros()
### IF USING ROOT TO PLOT - CAN SET GLUEX STYLE HERE ###
gInterpreter.ProcessLine('#include "gluex_style.h"')
gluex_style = ROOT.gluex_style() # Returns TStyle
gluex_style.SetPadRightMargin(0.08)
gluex_style.cd()

############## SET ALIASES ##############
THStack = ROOT.THStack
TCanvas = ROOT.TCanvas
RDataFrame = ROOT.RDataFrame
ROOT.ROOT.EnableImplicitMT()

############## PARSE COMMAND LINE ARGS ##############
fitName = f'{REPO_HOME}/gen_amp/result.fit'
hist_output_name = "histograms"; # -> histograms.pdf

results = FitResults( fitName )
if not results.valid():
    print(f'Invalid fit result in file: {fitName}')
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
    "Metapi": [ "Metapi", "MASS(ETA,PI0)", "M(#eta#pi)", 50, 1.04, 1.72, "" ],
    "Meta":   [ "Meta", "MASS(ETA)", "M(#eta)", 50, 0.49, 0.61, "" ],
    "Mpi0":   [ "Mpi0", "MASS(PI0)", "M(#pi^{0})", 50, 0.1, 0.18, "" ],
    "cosHel": [ "cosHel", "GJCOSTHETA(ETA,PI0,GLUEXBEAM)", "cos(#theta_{hel})", 50, -1, 1, "" ],

    # 2D Hists
    # HistName:     [ xname, xfunction, title, nx_bins, x_min, x_max, yname, yfunction, ny_bins, y_min, y_max, drawOptions]
    # "cosHelvsMass": [ "Metapi", "MASS(ETA,PI0)", "M(#eta#pi) vs cos(#theta_{hel})", 100, 1.04, 1.72, "cosHel", "GJCOSTHETA(ETA,PI0,GLUEXBEAM)", 100, -1, 1, "COLZ"],
}
N_BOOKED_HISTS = len(HISTS_TO_BOOK)

############## SETUP ##############
N_PARTICLES = 4
particles = ['GLUEXBEAM','PROTON','ETA','PI0']
kData, kBkgnd, kAccMC, kGenMC, kNumTypes = 0, 1, 2, 3, 4
kColors = {
    kData: ROOT.kBlack,
    kBkgnd: ROOT.kRed-9,
    kAccMC: ROOT.kGreen-8,
    kGenMC: ROOT.kAzure-4}  # for the 4 data sources
plotGen = PlotGenerator( results )
reactionName =  results.reactionList()[0]

HISTOGRAM_STORAGE = {} # {type: [hist1, hist2, ...]}
DRAW_OPT_STORAGE = {}
for type in [kData, kBkgnd, kAccMC, kGenMC]:

    ########### LOAD THE DATA ###########
    # Reaction: { Variable: [Values] } }
    value_map = plotGen.projected_values(reactionName, type, N_PARTICLES)
    value_map = value_map[reactionName][type]
    value_map = {k: np.array(v) for k,v in value_map}
    df = ROOT.RDF.MakeNumpyDataFrame(value_map)
    columns = df.GetColumnNames()

    ######### RESTRUCTURE DATA FOR NICER CALCULATIONS #########
    df.Define("GLUEXTARGET", "std::vector<float> p{0.938272, 0.0, 0.0, 0.0}; return p;")
    for i, particle in enumerate(particles):
        cmd = f"std::vector<float> p{{ PxP{i}, PyP{i}, PzP{i}, EnP{i} }}; return p;"
        df = df.Define(f"{particle}", cmd)
    # print(df.Describe())

    ######### BOOK HISTOGRAMS #########
    BOOKED_HISTOGRAMS, DRAW_OPTIONS = book_histogram(df, HISTS_TO_BOOK, columns)
    HISTOGRAM_STORAGE[type] = BOOKED_HISTOGRAMS
    DRAW_OPT_STORAGE[type] = DRAW_OPTIONS


###########################################################
### NOW CONFIGURE HOW YOU WANT TO DRAW THE HISTOGRAMS ####
### HERE IS AN EXAMPLE... BUT IMPOSSIBLE TO MAKE GENERIC #

ncols = int(np.floor(np.sqrt(len(HISTS_TO_BOOK))))
nrows = int(np.ceil(len(HISTS_TO_BOOK)/ncols))

canvas = TCanvas("canvas", "canvas", 1440, 1080)
canvas.Clear()
canvas.Divide(ncols, nrows)

canvas.Print(f"{hist_output_name}.pdf[")
stacked = [] # need to store them otherwise they get garbage collected
for ihist in range(N_BOOKED_HISTS):
    stacked.append(THStack("stack",""))
    for type in [kBkgnd, kAccMC]:
        booked_hist = HISTOGRAM_STORAGE[type][ihist]
        drawOptions = DRAW_OPT_STORAGE[type][ihist]
        booked_hist.SetFillColorAlpha(kColors[type],0.8)
        booked_hist.SetLineColor(0)
        stacked[ihist].Add(booked_hist.GetPtr())
    canvas.cd(ihist+1)
    stacked[ihist].Draw('HIST')
    HISTOGRAM_STORAGE[kData][ihist].Draw('E SAME')
canvas.Print(f"{hist_output_name}.pdf")
canvas.Print(f"{hist_output_name}.pdf]")
