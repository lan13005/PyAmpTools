#!/usr/bin/env python3

import ROOT
import os
from typing import List
import argparse

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
TH1                  = ROOT.TH1
TFile                = ROOT.TFile
AmpToolsInterface    = ROOT.AmpToolsInterface
FitResults           = ROOT.FitResults
ROOTDataReader       = ROOT.ROOTDataReader
Zlm                  = ROOT.Zlm
EtaPiPlotGenerator   = ROOT.EtaPiPlotGenerator
PlotGenerator        = ROOT.PlotGenerator
# PlotFactory          = ROOT.PlotFactory
# PlotterMainWindow    = ROOT.PlotterMainWindow


############## PARSE COMMANDLINE ARGUMENTS ##############
parser = argparse.ArgumentParser(description='Plotter')
parser.add_argument('-f', type=str, default='', help='Fit file name')
parser.add_argument('-o', type=str, default='plotter_results.root', help='Output file name')
parser.add_argument('-s', type=str, default='all', help='semicolon separated list of coherent sums to plot. Sums are underscore separated amplitudes. Empty string will plot sum of all waves')
parser.add_argument('-a', type=bool, default=True, help='Do acceptance correction or not')
parser.add_argument('-var', type=bool, default=False, help='Plot all variables or not')
parser.add_argument('-gen', type=bool, default=False, help='Plot gen histograms or not')
args = parser.parse_args()

fitName = args.f
ofile = args.o
ampString = args.s
keepAllAmps = ampString == 'all'
doAccCorr = args.a
plotAllVars = args.var
plotGenHists = args.gen

if plotAllVars: print(" >> Plotting all variables")
else: print(" >> Plotting only Mass variables")
if plotGenHists: print(" >> Plotting generated histograms")
else: print(" >> Not plotting generated histograms")
if doAccCorr: print(" >> Doing acceptance correction")
else: print(" >> Not doing acceptance correction")


############## LOAD FIT RESULTS ##############
if ampString == '': print(" >> No amplitude specified. Exiting..."); exit(1)
if fitName == '': print(" >> No fit file specified. Exiting..."); exit(1)
results = FitResults( fitName )
if not results.valid():
    print(f'Invalid fit result in file: {fitName}')
    exit()

############## REGISTER OBJECTS FOR AMPTOOLS ##############
AmpToolsInterface.registerAmplitude( Zlm() )
AmpToolsInterface.registerDataReader( ROOTDataReader() )


############### PLOTTING TIME! ################
print(" >> Loading FitResults into PlotGenerator...")
opt = PlotGenerator.Option
if plotGenHists: opt = PlotGenerator.kDefault
else: opt = PlotGenerator.kNoGenMC
plotGen = EtaPiPlotGenerator( results )

if keepAllAmps: print( "\n >> Plotting all amplitudes\n" )

wavesets = ampString.split(';')
for waveset in wavesets:
    amp_map = {}
    print(f' >> Plotting waveset: {waveset}')
    print(f' >>   Only these amplitude contributions:')
    waves = waveset.split('_')
    for wave in waves:
        amp_map[wave] = -1 # Placeholder
        print(f' >>     {wave}')
    print('--------------------')

    outName = f'plotter_{waveset}.root'
    print(f' >> Output file: {outName}')

    plotfile = TFile(outName, 'RECREATE') # : TFile
    TH1.AddDirectory(False)

    reactionList = results.reactionList() # vector<string>
    for reaction in reactionList:
        plotGen.enableReaction(reaction)
    sums = plotGen.uniqueSums() # vector<string>
    amps = plotGen.uniqueAmplitudes() # vector<string>
    print(f' >>   All possible sums:')
    for sum in sums:
        print(f' >>     {sum}')
    print(f' >>   All possible amplitudes:')
    for i, amp in enumerate(amps):
        print(f' >>     {amp}', end='')
        if amp in amp_map or keepAllAmps:
            print(' - saving this one')
            amp_map[amp] = i
    print()

    print(" >> Setting ampliudes/sums to use")
    if not keepAllAmps:
        for i in range(len(amps)):
            plotGen.disableAmp(i)
        for k,v in amp_map.items():
            print(f' >> Enabling only {k}')
            if v == -1: print(f' >> WARNING: Amplitude {k} not found in fit results. Exiting...'); exit(1)
            plotGen.enableAmp(v)

    for isum in range(len(sums)+1): # loop over sums, last one combines them all
        for i in range(len(sums)): # renable everything
            plotGen.enableSum(i)
        if isum < len(sums): # disable all sums except the one we want
            for i in range(len(sums)):
                if i != isum:
                    plotGen.disableSum(i)

        ## Loop of data, accmc, genmc, bkgnd
        print(f' >> PlotGenerator has {PlotGenerator.kNumTypes} data source. Generating up to {EtaPiPlotGenerator.kNumHists} histograms each')
        for iplot in range(PlotGenerator.kNumTypes):
            if isum < sums.size() and iplot == PlotGenerator.kData: continue # only plot data once
            for ivar in range(EtaPiPlotGenerator.kNumHists):
                for ireact, reactionName in enumerate(reactionList):
                    histName = reactionName+'_'
                    if not plotAllVars and ivar > 3: continue # only plot mass variables (see EtaPiPlotGenerator.h for enumeration choice)
                    if not plotGenHists and iplot == PlotGenerator.kGenMC: continue # only plot generated histograms if specified

                    ## Set unique histogram names for each plot
                    if ivar == EtaPiPlotGenerator.kEtaPiMass: histName += "Metapi"
                    elif ivar == EtaPiPlotGenerator.kEtaPiMass_40MeVBin: histName += "Metapi_40MeVBin"
                    elif ivar == EtaPiPlotGenerator.kEtaProtonMass: histName += "Metaproton"
                    elif ivar == EtaPiPlotGenerator.kPi0ProtonMass: histName += "Mpi0proton"
                    elif ivar == EtaPiPlotGenerator.kEtaCosTheta: histName += "cosTheta"
                    elif ivar == EtaPiPlotGenerator.kPhiPi: histName += "phiPi"
                    elif ivar == EtaPiPlotGenerator.kPhiEta: histName += "phiEta"
                    elif ivar == EtaPiPlotGenerator.kPhi: histName += "Phi"
                    elif ivar == EtaPiPlotGenerator.kphi: histName += "phi"
                    elif ivar == EtaPiPlotGenerator.kPsi: histName += "Psi"
                    elif ivar == EtaPiPlotGenerator.kt: histName += "t"
                    else: print(" >> WARNING: Unknown variable. Exiting..."); exit(1)

                    if iplot == PlotGenerator.kData: histName += "dat"
                    if iplot == PlotGenerator.kBkgnd: histName += "bkg"
                    if iplot == PlotGenerator.kAccMC: histName += "acc"
                    if iplot == PlotGenerator.kGenMC: histName += "gen"

                    if isum < len(sums):
                        histName += f'_{sums[isum]}'

                    # Write histogram to file
                    hist = plotGen.projection(ivar, reactionName, iplot)
                    thist = hist.toRoot()
                    if thist.Integral() != 0: # only write if there are entries
                        thist.SetName(histName)
                        plotfile.cd()
                        thist.Write()
                    else:
                        print(f" >> WARNING: Histogram {histName} has no entries. Not writing to file!")

    plotfile.Close()

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
