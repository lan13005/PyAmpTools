#!/usr/bin/env python3

import os
import argparse
import re
import sys
import atiSetup

def extract_ff(results, outfileName='', acceptanceCorrect=True, fmt='.5f', regex_merge=None):
    '''
    Extract Fit Fractions and phase differences between pairs of waves from a FitResults object

    Regex merge is a useful tool to merge amplitudes that are related to each other (user-specified)
        For example waveset: D-2- D-1- D0- D1- D2- D-2+ D-1+ D0+ D1+ D2+
        To remove the sign at the end (merge reflectivites)     use = r'[-+]$'
        To remove first sign and number (merge M-projections)   use = r'[-+]?(\d+)'
        To remove numbers and signs to merge all D:             use = r'[-+]?(\d+)[-+]'

    Args:
        results (FitResults): FitResults object
        outfileName (str): Output root file name or dump to stdout if ''
        acceptanceCorrect (bool): Acceptance correct the values
        fmt (str): string format for printing
        regex_merge (str): Merge amplitudes: Regex pair (pattern, replace) separated by ~>

    Returns:
        None, dumps a file to outfileName or stdout
    '''

    def write_ff(amp, intensity, error):
        outfile.write(f'FIT FRACTION {amp} = {intensity/total_intensity:{fmt}} +/- {error/total_intensity:{fmt}}\n')

    ############### PLOTTING TIME! ################
    outfile = open(outfileName, 'w') if outfileName != '' else sys.stdout
    total_intensity, total_error = results.intensity(acceptanceCorrect)
    outfile.write(f"TOTAL EVENTS = {total_intensity} +/- {total_error}\n")
    uniqueAmps = results.ampList() # vector<string>
    uniqueAmps = [str(amp) for amp in uniqueAmps]

    ######## DETERMINE UNIQUE AMPLITUDES AND PLOT THEM ALL #########
    print('\nAll Unique Amplitudes:')
    for amp in uniqueAmps: # amp ~ "Reaction::Sum::Amp" whereas amp ~ "Amp"
        amp = str(amp)#.split('::')[-1] # amp is of type TString I think, convert first
        # Print all amplitudes including including constrained ones, polarizations, etc
        useamp = [amp]
        print(f' -> {amp}')
        intensity, error = results.intensity(useamp, acceptanceCorrect)
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
                intensity, error = results.intensity(amps, acceptanceCorrect)
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

    if outfileName != '':
        outfile.close()

if __name__ == '__main__':

    ############## SET ENVIRONMENT VARIABLES ##############
    REPO_HOME     = os.environ['REPO_HOME']

    ################### LOAD LIBRARIES ##################
    atiSetup.setup(globals())

    ############## PARSE COMMANDLINE ARGUMENTS ##############
    parser = argparse.ArgumentParser(description='Extract Fit Fractions from FitResults')
    parser.add_argument('fitName', type=str, default='', help='Fit file name')
    parser.add_argument('--outputfileName', type=str, default='', help='Output file name')
    parser.add_argument('--a', type=bool, default=True, help='Acceptance correct values')
    parser.add_argument('--fmt', type=str, default='.5f', help='Format string for printing')
    parser.add_argument('--regex_merge', type=str, nargs='+', help='Merge amplitudes: Regex pair (pattern, replace) separated by ~>')
    args = parser.parse_args(sys.argv[1:])

    ############## LOAD FIT RESULTS ##############
    fitName = args.fitName
    outfileName = args.outputfileName
    acceptanceCorrect = args.a
    fmt=args.fmt
    regex_merge = args.regex_merge
    assert( os.path.isfile(fitName) ), f'Fit file does not exist at specified path'

    ############## LOAD FIT RESULTS OBJECT ##############
    results = FitResults( fitName )
    if not results.valid():
        print(f'Invalid fit result in file: {fitName}')
        exit()

    ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude( Zlm() )
    AmpToolsInterface.registerDataReader( DataReader() )

    ############## EXTRACT FIT FRACTIONS ##############
    extract_ff(results, outfileName, acceptanceCorrect, fmt, regex_merge)
