#!/usr/bin/env python3

import ROOT
import os
from typing import List
import argparse

def main():
    ############## SET ENVIRONMENT VARIABLES ##############
    REPO_HOME     = os.environ['REPO_HOME']

    ############## LOAD LIBRARIES ##############
    ROOT.gSystem.Load('libAmps.so')
    ROOT.gSystem.Load('libAmpPlotter.so')
    ROOT.gSystem.Load('libDataIO.so')
    ROOT.gSystem.Load('libAmpTools.so')

    # Dummy functions that just prints initialization
    #  This is to make sure the libraries are loaded
    #  as python is interpreted
    ROOT.initializeAmps(True)
    ROOT.initializeDataIO(True)

    ################ SET ALIAS ###################
    AmpToolsInterface    = ROOT.AmpToolsInterface
    Zlm                  = ROOT.Zlm
    ROOTDataReader       = ROOT.ROOTDataReader
    EtaPiPlotGenerator   = ROOT.EtaPiPlotGenerator
    FitResults           = ROOT.FitResults
    PlotFactory          = ROOT.PlotFactory
    PlotterMainWindow    = ROOT.PlotterMainWindow

    ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude( Zlm() )
    AmpToolsInterface.registerDataReader( ROOTDataReader() )

    parser = argparse.ArgumentParser()
    # require argparse to take on argument
    parser.add_argument('fitName', type=str, help='Fit Result File')
    args = parser.parse_args()
    fitName = args.fitName

    results = FitResults( fitName )

    if not results.valid():
        print(f'Invalid fit result in file: {fitName}')
        exit()

    plotGen = EtaPiPlotGenerator( results )


    #################################################################################
    ## CANT GET PLOTTER GUI TO WORK, MULTIPLE INSTANCES OF TAPPLICATION LAUNCHING ###
    #################################################################################
    # print(" >> Plot generator ready, starting GUI...")

    # app = ROOT.TApplication("app", ROOT.nullptr, ROOT.nullptr)

    # ROOT.gStyle.SetFillColor(10)
    # ROOT.gStyle.SetCanvasColor(10)
    # ROOT.gStyle.SetPadColor(10)
    # ROOT.gStyle.SetFillStyle(1001)
    # ROOT.gStyle.SetPalette(1)
    # ROOT.gStyle.SetFrameFillColor(10)
    # ROOT.gStyle.SetFrameFillStyle(1001)

    # print(" >> Creating plotter GUI...")
    # factory = PlotFactory( plotGen )
    # print(" >> Plotter GUI ready, starting main window...")
    # PlotterMainWindow( ROOT.gClient.GetRoot(), factory );
    # print(" >> Main window ready, starting event loop...")
    # app.Run()
    #################################################################################
