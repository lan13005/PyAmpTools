#!/usr/bin/env python3

import ROOT
import os
from typing import List
import argparse
from pyamptools import atiSetup

def PlotGen(
        results,
        ofile,
        ampString = 'all',
        keepAllAmps = False,
        plotAllVars = False,
        plotGenHists = False,
        doAccCorr = True,
    ):
    '''
    Generates plots from a fit result file. The plots are saved to a root file.
        A c++ user-subclassed PlotGenerator object (here, EtaPiPlotGenerator) is used to generate the plots. Histogram names, ranges, kinematic
        quantities are all defined there.

    Args:
        results (FitResults): FitResults object containing the fit results.
        ofile (str): name of output root file.
        ampString (str): Semicolon separated list of coherent sums to plot. Sums are underscore separated amplitudes. Empty string will plot sum of all waves.
        keepAllAmps (bool): If True, all amplitudes are summed and plotted. If False, only the amplitudes specified in ampString will be plotted.
        plotAllVars (bool): If True, all variables will be plotted.
        plotGenHists (bool): If True, generated histograms will be plotted.
        doAccCorr (bool): If True, acceptance correction will be applied to the histograms.

    Returns:
        None, but saves drawn amplitufe weighted histograms to a root file.
    '''

    plotfile = TFile(ofile, 'RECREATE')
    TH1.AddDirectory(False)

    if plotAllVars: print(" >> Plotting all variables")
    else: print(" >> Plotting only Mass variables")
    if plotGenHists: print(" >> Plotting generated histograms")
    else: print(" >> Not plotting generated histograms")
    if doAccCorr: print(" >> Doing acceptance correction")
    else: print(" >> Not doing acceptance correction")
    if keepAllAmps: print( "\n >> Plotting all amplitudes\n" )

    plotGen = EtaPiPlotGenerator( results )

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

        print(f' >> Output file: {ofile}')

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
            print(f' >> PlotGenerator has {EtaPiPlotGenerator.kNumTypes} data source. Generating up to {EtaPiPlotGenerator.kNumHists} histograms each')
            for iplot in range(PlotGenerator.kNumTypes):
                if isum < sums.size() and iplot == EtaPiPlotGenerator.kData: continue # only plot data once
                for ivar in range(EtaPiPlotGenerator.kNumHists):
                    for ireact, reactionName in enumerate(reactionList):
                        # print(f' >> Plotting reaction {reactionName}, histogram {ivar}, data source {iplot}')
                        histName = reactionName+'_'
                        if not plotAllVars and ivar > 3: continue # only plot mass variables (see EtaPiPlotGenerator.h for enumeration choice)
                        if not plotGenHists and iplot == EtaPiPlotGenerator.kGenMC: continue # only plot generated histograms if specified
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

                        if iplot == EtaPiPlotGenerator.kData: histName += "dat"
                        if iplot == EtaPiPlotGenerator.kBkgnd: histName += "bkg"
                        if iplot == EtaPiPlotGenerator.kAccMC: histName += "acc"
                        if iplot == EtaPiPlotGenerator.kGenMC: histName += "gen"

                        if isum < len(sums):
                            histName += f'_{sums[isum]}'

                        # print(f' >> Plotting histogram {histName} for reaction {reactionName} with data source {iplot}')
                        # Write histogram to file
                        hist = plotGen.projection(ivar, reactionName, iplot)
                        thist = hist.toRoot()
                        if thist.Integral() != 0: # only write if there are entries
                            print(f" >> Writing histogram {histName} to file")
                            thist.SetName(histName)
                            plotfile.cd()
                            thist.Write()
                        else:
                            print(f" >> WARNING: Histogram {histName} has no entries. Not writing to file!")

        plotfile.Close()


def _cli_plotgen():
    ''' Command line interface for generating plots from a fit result file. The plots are saved to a root file. '''

    ############## SET ENVIRONMENT VARIABLES ##############
    REPO_HOME     = os.environ['REPO_HOME']

    ################### LOAD LIBRARIES ##################
    atiSetup.setup(globals())

    ############## PARSE COMMANDLINE ARGUMENTS ##############
    parser = argparse.ArgumentParser(description='Plotter')
    parser.add_argument('fitName', type=str, help='Fit file name')
    parser.add_argument('-o', type=str, default='plotter_results.root', help='Output file name')
    parser.add_argument('-s', type=str, default='all', help='semicolon separated list of coherent sums to plot. Sums are underscore separated amplitudes. Empty string will plot sum of all waves')
    parser.add_argument('-a', type=bool, default=True, help='Do acceptance correction or not')
    parser.add_argument('-var', type=bool, default=False, help='Plot all variables or not')
    parser.add_argument('-gen', type=bool, default=False, help='Plot gen histograms or not')
    args = parser.parse_args()

    fitName = args.fitName
    ofile = args.o
    ampString = args.s
    keepAllAmps = ampString == 'all'
    doAccCorr = args.a
    plotAllVars = args.var
    plotGenHists = args.gen
    assert( os.path.isfile(fitName) ), f'Fit file does not exist at specified path'

    ############## LOAD FIT RESULTS ##############
    if ampString == '': print(" >> No amplitude specified. Exiting..."); exit(1)
    if fitName == '': print(" >> No fit file specified. Exiting..."); exit(1)
    results = FitResults( fitName )
    if not results.valid():
        print(f'Invalid fit result in file: {fitName}')
        exit()

    ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude( Zlm() )
    AmpToolsInterface.registerDataReader( DataReader() )

    ############### PLOTTING TIME! ################
    print(" >> Loading FitResults into PlotGenerator...")
    # opt = PlotGenerator.Option
    # if plotGenHists: opt = PlotGenerator.kDefault
    # else: opt = PlotGenerator.kNoGenMC

    PlotGen(results, ofile, ampString, keepAllAmps, plotAllVars, plotGenHists, doAccCorr)

    #################################################################################
    ## CANT GET PLOTTER GUI TO WORK, MULTIPLE INSTANCES OF TAPPLICATION LAUNCHING ###
    #################################################################################

    ################ SET ADDITIONAL ALIAS ###################
    # PlotFactory          = ROOT.PlotFactory
    # PlotterMainWindow    = ROOT.PlotterMainWindow

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

if __name__ == '__main__':
    _cli_plotgen()
