#!/usr/bin/env python3

import ROOT
import os
from typing import List
import argparse
import re
from utils import check_nvidia_devices

############## SET ENVIRONMENT VARIABLES ##############
REPO_HOME     = os.environ['REPO_HOME']
os.environ['ATI_USE_MPI'] = "0" # set to 1 to use MPI libraries
os.environ['ATI_USE_GPU'] = "0"
from atiSetup import *

################ SET ADDITIONAL ALIAS ###################

############## PARSE COMMANDLINE ARGUMENTS ##############
parser = argparse.ArgumentParser(description='Extract Fit Fractions from FitResults')
parser.add_argument('fitName', type=str, default='', help='Fit file name')
parser.add_argument('ofileName', type=str, default='', help='Output file name')
parser.add_argument('-a', type=bool, default=True, help='Acceptance correct values')
parser.add_argument('-fmt', type=str, default='.5f', help='Format string for printing')
parser.add_argument('-regex_merge', type=str, nargs='+', help='Merge amplitudes: Regex pair (pattern, replace) separated by ~>')
args = parser.parse_args()

############## LOAD FIT RESULTS ##############
fitName = args.fitName
outfilename = args.ofileName
acceptance = args.a
fmt=args.fmt
regex_merge = args.regex_merge
assert( os.path.isfile(fitName) ), f'Fit file does not exist at specified path'

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
AmpToolsInterface.registerDataReader( DataReader() )


############### PLOTTING TIME! ################
print(" >> Loading FitResults into PlotGenerator...")
plotGen = EtaPiPlotGenerator( results )

outfile = open(outfilename, 'w')
total_intensity, total_error = results.intensity(acceptance)
outfile.write(f"TOTAL EVENTS = {total_intensity} +/- {total_error}\n")
uniqueAmps = plotGen.fullAmplitudes() # vector<string>
uniqueAmps = [str(amp) for amp in uniqueAmps]

def write_ff(amp, intensity, error):
    outfile.write(f'FIT FRACTION {amp} = {intensity/total_intensity:{fmt}} +/- {error/total_intensity:{fmt}}\n')

######## DETERMINE UNIQUE AMPLITUDES AND PLOT THEM ALL #########
print('\nAll Unique Amplitudes:')
for amp in uniqueAmps: # amp ~ "Reaction::Sum::Amp" whereas amp ~ "Amp"
    amp = str(amp)#.split('::')[-1] # amp is of type TString I think, convert first
    # Print all amplitudes including including constrained ones, polarizations, etc
    useamp = [amp]
    print(f' -> {amp}')
    intensity, error = results.intensity(useamp, acceptance)
    write_ff(amp, intensity, error)

######## MERGE AMPLITUDES STRIPPING REGEX MATCHED STRING #########
if regex_merge is not None:
    for regex in regex_merge:
        pattern, replace = regex.split('~>') if '~>' in regex else (regex, '')
        pattern, replace = r""+pattern, r""+replace
        print(f"\nMerged Amplitude Groups based on regex sub: r'{pattern}' -> r'{replace}':")
        merged = {} # dictionary of lists
        for amps in uniqueAmps:
            filterd_amp = re.sub(pattern, replace, amps) # regex to remove numbers, r"" force conversion to raw string
            if filterd_amp not in merged:
                merged[filterd_amp] =[amps]
            else:
                merged[filterd_amp].append(amps)
        for merged_amp, amps in merged.items():
            print(f' -> {merged_amp}')
            for amp in amps: print(f'     {amp}')
        for merged_amp, amps in merged.items():
            intensity, error = results.intensity(amps, acceptance)
            write_ff(merged_amp, intensity, error)
        merged.clear()

######### WRITE ALL POSSIBLE PHASE DIFFERENCES ##########
for amp1 in uniqueAmps:
    for amp2 in uniqueAmps:
        amp1, amp2 = str(amp1), str(amp2) # amps are TStrings
        if amp1 == amp2: continue
        same_reaction = amp1.split("::")[0] == amp2.split("::")[0]
        same_sum      = amp1.split("::")[1] == amp2.split("::")[1]
        if not same_reaction or not same_sum: continue # interfence only in same {reaction, sum}
        phase, error = results.phaseDiff(amp1, amp2)
        outfile.write(f'PHASE DIFFERENCE {amp1} {amp2} = {phase:{fmt}} +/- {error:{fmt}}\n')
