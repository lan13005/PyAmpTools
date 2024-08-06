#!/usr/bin/env python3

import argparse
import os
import re
import sys


def extract_ff(results, outfileName="", fmt=".5f", regex_merge=None, no_phases=False, only=None):
    """
    Extract Fit Fractions and phase differences between pairs of waves from a FitResults object

    Regex merge can be a useful tool to merge amplitudes that are related to each other (user-specified)
        For example waveset: D-2- D-1- D0- D1- D2- D-2+ D-1+ D0+ D1+ D2+
        To remove the sign at the end (merge reflectivites)     use = r'[-+]$'
        To remove first sign and number (merge M-projections)   use = r'[-+]?(\d+)'
        To remove numbers and signs to merge all D:             use = r'[-+]?(\d+)[-+]'

    Args:
        results (FitResults): FitResults object
        outfileName (str): Output root file name or dump to stdout if ''
        acceptanceCorrect (bool): Acceptance correct the values
        fmt (str): string format for printing
        regex_merge (List[str]): Merge amplitudes: List of Regex pairs (pattern, replace) separated by ~>
        only (str): Only dump fit fractions for ["acc", "noacc"]. Default dumps FF concatenated by "|"

    Returns:
        None, dumps a file to outfileName or stdout
    """

    def write_ff(amp, intensity, error, intensity_corr, error_corr, only=None):
        if only is None:
            outfile.write(f"FIT FRACTION {amp} = {intensity_corr/total_intensity_corr:{fmt}}|{intensity/total_intensity:{fmt}} +- {error_corr/total_intensity_corr:{fmt}}|{error/total_intensity:{fmt}}\n")
        elif only == "acc":
            outfile.write(f"FIT FRACTION {amp} = {intensity/total_intensity:{fmt}} +- {error/total_intensity:{fmt}}\n")
        elif only == "noacc": # no accepatance = correct for acceptance
            outfile.write(f"FIT FRACTION {amp} = {intensity_corr/total_intensity_corr:{fmt}} +- {error_corr/total_intensity_corr:{fmt}}\n")
        

    ############### LOAD RESULTS TIME! ################
    outfile = open(outfileName, "w") if outfileName != "" else sys.stdout
    total_intensity, total_error = results.intensity(False)
    total_intensity_corr, total_error_corr = results.intensity(True)
    outfile.write("########################################################################\n")
    outfile.write("# Values on the left of | are acceptance corrected, on the right are not\n")
    outfile.write("########################################################################\n\n")
    if only is None:
        outfile.write(f"TOTAL EVENTS = {total_intensity_corr:0.2f}|{total_intensity:0.2f} +- {total_error_corr:0.2f}|{total_error:0.2f}\n")
    elif only == "acc":
        outfile.write(f"TOTAL EVENTS = {total_intensity:0.2f} +- {total_error:0.2f}\n")
    elif only == "noacc":
        outfile.write(f"TOTAL EVENTS = {total_intensity_corr:0.2f} +- {total_error_corr:0.2f}\n")
    else:
        raise ValueError(f"Invalid 'only' argument: {only}. Expected either ['acc', 'noacc']")

    uniqueAmps = results.ampList()  # vector<string>
    uniqueAmps = [str(amp) for amp in uniqueAmps]

    ######## DETERMINE UNIQUE AMPLITUDES AND PLOT THEM ALL #########
    print("\nAll Unique Amplitudes:")
    for amp in uniqueAmps:  # amp ~ "Reaction::Sum::Amp" whereas amp ~ "Amp"
        amp = str(amp)  # .split('::')[-1] # amp is of type TString I think, convert first
        # Print all amplitudes including including constrained ones, polarizations, etc
        useamp = [amp]
        print(f" -> {amp}")
        intensity, error = results.intensity(useamp, False)
        intensity_corr, error_corr = results.intensity(useamp, True)
        write_ff(amp, intensity, error, intensity_corr, error_corr, only)

    ######## MERGE AMPLITUDES STRIPPING REGEX MATCHED STRING #########
    if regex_merge is not None:
        for regex in regex_merge:
            pattern, replace = regex.split("~>") if "~>" in regex else (regex, "")
            pattern, replace = r"" + pattern, r"" + replace
            print(f"\nMerged Amplitude Groups based on regex sub: r'{pattern}' -> r'{replace}':")
            merged = {}  # dictionary of lists
            for amps in uniqueAmps:
                if re.search(pattern, amps) is None:
                    continue
                filterd_amp = re.sub(pattern, replace, amps)  # regex to remove numbers, r"" force conversion to raw string
                if filterd_amp not in merged:
                    merged[filterd_amp] = [amps]
                else:
                    merged[filterd_amp].append(amps)

            for merged_amp, amps in merged.items():
                # if len(amps) <= 1:
                #     continue  # skip if none merged. Turn off since if Pmag = 1 then PosIm will not exist (only PosRe)
                print(f" -> {merged_amp} merged {len(amps)} amplitudes:")
                for amp in amps:
                    print(f"     {amp}")

                intensity, error = results.intensity(amps, False)
                intensity_corr, error_corr = results.intensity(amps, True)
                write_ff(merged_amp, intensity, error, intensity_corr, error_corr, only)
            merged.clear()

    ######### WRITE ALL POSSIBLE PHASE DIFFERENCES ##########
    if not no_phases:
        for amp1 in uniqueAmps:
            for amp2 in uniqueAmps:
                amp1, amp2 = str(amp1), str(amp2)  # amps are TStrings
                if amp1 == amp2:
                    continue
                same_reaction = amp1.split("::")[0] == amp2.split("::")[0]
                same_sum = amp1.split("::")[1] == amp2.split("::")[1]
                if not same_reaction or not same_sum:
                    continue  # interfence only in same {reaction, sum}
                phase, error = results.phaseDiff(amp1, amp2)
                outfile.write(f"PHASE DIFFERENCE {amp1} {amp2} = {phase:{fmt}} +- {error:{fmt}}\n")

    outfile.write("\n########### FIT STATUS ###########\n")
    outfile.write(f"# bestMinimum = {results.bestMinimum()}\n")
    outfile.write(f"# lastMinuitCommandStatus = {results.lastMinuitCommandStatus()}\n")
    outfile.write(f"# eMatrixStatus = {results.eMatrixStatus()}\n")

    if outfileName != "":
        outfile.close()


def _cli_extract_ff():
    """Command line interface for extracting fit fractions from an amptools fit results"""

    ############## PARSE COMMANDLINE ARGUMENTS ##############
    parser = argparse.ArgumentParser(description="Extract Fit Fractions from FitResults")
    parser.add_argument("fitFile", type=str, default="", help="Amptools FitResults file name")
    parser.add_argument("--outputfileName", type=str, default="extracted_fitfracs.txt", help="Output file name")
    parser.add_argument("--fmt", type=str, default=".5f", help="Format string for printing")
    parser.add_argument("--regex_merge", type=str, nargs="+", help="Merge amplitudes: Regex pair (pattern, replace) separated by ~>")
    parser.add_argument("--no_phases", action="store_true", help="Do not dump phase differences")
    parser.add_argument("--only", type=str, default=None, help="Only dump fit fractions for ['acc', 'noacc']")
    args = parser.parse_args(sys.argv[1:])

    ################### LOAD LIBRARIES ##################
    from pyamptools import atiSetup

    atiSetup.setup(globals())

    ############## LOAD FIT RESULTS ##############
    fitFile = args.fitFile
    outfileName = args.outputfileName
    fmt = args.fmt
    regex_merge = args.regex_merge
    no_phases = args.no_phases
    only = args.only
    assert os.path.isfile(fitFile), "Fit Results file does not exist at specified path"

    ############## LOAD FIT RESULTS OBJECT ##############
    results = FitResults(fitFile)
    if not results.valid():
        print(f"Invalid fit result in file: {fitFile}")
        exit()

    ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude(Zlm())
    AmpToolsInterface.registerDataReader(DataReader())

    ############## EXTRACT FIT FRACTIONS ##############
    extract_ff(results, outfileName, fmt, regex_merge, no_phases, only)


if __name__ == "__main__":
    _cli_extract_ff()
