#!/usr/bin/env python3

import ROOT
import os
from typing import List
import argparse
import re

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


############## PARSE COMMANDLINE ARGUMENTS ##############
parser = argparse.ArgumentParser(description='Extract Fit Fractions from FitResults')
parser.add_argument('-f', type=str, default='', help='Fit file name')
parser.add_argument('-o', type=str, default='', help='Output file name')
parser.add_argument('-a', type=bool, default=True, help='Acceptance correct values')
parser.add_argument('-fmt', type=str, default='.5f', help='Format string for printing')
parser.add_argument('-regex_merge', type=str, nargs='+', help='Merge amplitudes under this regex pattern')
args = parser.parse_args()

############## LOAD FIT RESULTS ##############
fitName = args.f
outfilename = args.o
acceptance = args.a
fmt=args.fmt
regex_merge = args.regex_merge
'''
Regex merge is a useful tool to merge amplitudes that are related to each other (user-specified)
For example waveset: D-2- D-1- D0- D1- D2- D-2+ D-1+ D0+ D1+ D2+
To remove the sign at the end (merge reflectivites)     use = r'[-+]$'
To remove first sign and number (merge M-projections)   use = r'[-+]?(\d+)'
To remove numbers and signs to merge all D:             use = r'[-+]?(\d+)[-+]'
'''
results = FitResults( fitName )
if not results.valid():
    print(f'Invalid fit result in file: {fitName}')
    exit()

############## REGISTER OBJECTS FOR AMPTOOLS ##############
AmpToolsInterface.registerAmplitude( Zlm() )
AmpToolsInterface.registerDataReader( ROOTDataReader() )


############### PLOTTING TIME! ################
print(" >> Loading FitResults into PlotGenerator...")
plotGen = EtaPiPlotGenerator( results )

outfile = open(outfilename, 'w')
total_intensity, total_error = results.intensity(acceptance)
outfile.write(f"TOTAL EVENTS = {total_intensity} +/- {total_error}\n")
fullamps = plotGen.fullAmplitudes() # vector<string>
uniqueAmps = {} # dictionary of lists

def write_ff(amp, intensity, error):
    outfile.write(f'FIT FRACTION {amp} = {intensity/total_intensity:{fmt}} +/- {error/total_intensity:{fmt}}\n')

######## DETERMINE UNIQUE AMPLITUDES AND PLOT THEM ALL #########
for i, fullamp in enumerate(fullamps): # fullamp ~ "Reaction::Sum::Amp" whereas amp ~ "Amp"
    amp = str(fullamp).split('::')[-1] # amp is of type TString I think, convert first
    if amp not in uniqueAmps:
        uniqueAmps[amp] = [fullamps[i]]
    else:
        uniqueAmps[amp].append(fullamps[i])
    # Plot all amplitudes including including constrained ones, polarizations, etc
    useamp = [fullamp]
    print(f' -> {fullamp}')
    intensity, error = results.intensity(useamp, acceptance)
    write_ff(amp, intensity, error)

######### PRINT UNIQUE AMPLITUDES (SUM OVER REACTION / POL) ##########
print("\nUnique Amplitude Groups Found:")
for uniqueAmp, amps in uniqueAmps.items():
    print(f' -> {uniqueAmp}')
    for amp in amps:
        print(f'     {amp}')
for uniqueAmp, amps in uniqueAmps.items():
    intensity, error = results.intensity(amps, acceptance)
    write_ff(uniqueAmp, intensity, error)

######## MERGE AMPLITUDES STRIPPING REGEX MATCHED STRING #########
if regex_merge is not None:
    for regex in regex_merge:
        print(f"\nMerged Amplitude Groups based on regex: r'{regex}':")
        merged = {} # dictionary of lists
        for uniqueAmp, amps in uniqueAmps.items():
            filterd_amp = re.sub(r""+regex, '', uniqueAmp) # regex to remove numbers, r"" force conversion to raw string
            if filterd_amp not in merged:
                merged[filterd_amp] = amps
            else:
                merged[filterd_amp].extend(amps)
        for merged_amp, amps in merged.items():
            print(f' -> {merged_amp}')
            for amp in amps: print(f'     {amp}')
        for merged, amps in merged.items():
            intensity, error = results.intensity(amps, acceptance)
            write_ff(merged, intensity, error)


######### WRITE ALL POSSIBLE PHASE DIFFERENCES ##########
for amp1 in fullamps:
    for amp2 in fullamps:
        amp1, amp2 = str(amp1), str(amp2) # amps are TStrings
        if amp1 == amp2: continue
        same_reaction = amp1.split("::")[0] == amp2.split("::")[0]
        same_sum      = amp1.split("::")[1] == amp2.split("::")[1]
        if not same_reaction or not same_sum: continue # interfence only in same {reaction, sum}
        phase, error = results.phaseDiff(amp1, amp2)
        outfile.write(f'PHASE DIFFERENCE {amp1} {amp2} = {phase:{fmt}} +/- {error:{fmt}}\n')
