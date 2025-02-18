#!/usr/bin/env python3

import argparse
import os
import re
import sys
from pyamptools.utility.general import load_yaml
from omegaconf.dictconfig import DictConfig

def extract_ff(results, outfileName="", fmt=".5f", test_regex=False, no_phases=False, only=None, regex_merge=None, yaml_file=''):
    """
    Extract fit fractions and phase differences between pairs of waves from a FitResults object.

    Args:
        results (FitResults): FitResults object containing fit results
        outfileName (str): Output root file name or dump to stdout if empty string
        fmt (str): String format for printing numbers
        test_regex (bool): If True, only test and print regex grouping without calculating intensities
        no_phases (bool): If True, skip calculating phase differences
        only (str): Only dump fit fractions for "acc" or "noacc". Default dumps both.
        regex_merge (List[str]): List of regex pattern/replace pairs for merging amplitudes.
            Pairs are separated by ~>. The substitution happens for all amplitude names.
            All amplitudes with same reduced name will be grouped into a list and a combined fit fraction
            calculated. See AmpTools' FitResults.intensity method.
            Examples:
                - '.*::(.*)::.*~>\\1': Captures text between :: and replaces full match
                - '.*(.)$~>\\1': Captures last character and replaces full match
                - '.*reaction_(000|045|090|135)::(Pos|Neg)(?:Im|Re)::': Removes matched pattern,
                  allowing grouping over polarizations and mirrored sums
        yaml_file (str): YAML file name. Allows program to load YAML state, i.e. for getting user requested coherent sums
    """

    def write_ff(amp, intensity, error, intensity_corr, error_corr, only=None):
        if only is None:
            outfile.write(f"FIT FRACTION {amp} = {intensity_corr / total_intensity_corr:{fmt}}|{intensity / total_intensity:{fmt}} +- {error_corr / total_intensity_corr:{fmt}}|{error / total_intensity:{fmt}}\n")
        elif only == "acc":
            outfile.write(f"FIT FRACTION {amp} = {intensity / total_intensity:{fmt}} +- {error / total_intensity:{fmt}}\n")
        elif only == "noacc":  # no accepatance = correct for acceptance
            outfile.write(f"FIT FRACTION {amp} = {intensity_corr / total_intensity_corr:{fmt}} +- {error_corr / total_intensity_corr:{fmt}}\n")

    ############### LOAD RESULTS TIME! ################
    if not test_regex:
        outfile = open(outfileName, "w") if outfileName != "" else sys.stdout
        total_intensity, total_error = results.intensity(False)
        total_intensity_corr, total_error_corr = results.intensity(True)
        outfile.write("########################################################################\n")
        outfile.write("# Values on the left of | are acceptance corrected, on the right are not\n")
        outfile.write(f"# bestMinimum = {results.bestMinimum()}\n")
        outfile.write(f"# lastMinuitCommandStatus = {results.lastMinuitCommandStatus()}\n")
        outfile.write(f"# eMatrixStatus = {results.eMatrixStatus()}\n")
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
    if test_regex:
        print("\nAll Unique Amplitudes:")
        for amp in uniqueAmps:
            print(f" -> {amp}")
    elif not test_regex:
        print("\nAll Unique Amplitudes:")
        for amp in uniqueAmps:  # amp ~ "Reaction::Sum::Amp" whereas amp ~ "Amp"
            amp = str(amp)  # .split('::')[-1] # amp is of type TString I think, convert first
            # Print all amplitudes including including constrained ones, polarizations, etc
            useamp = [amp]
            print(f" -> {amp}")
            intensity, error = results.intensity(useamp, False)
            intensity_corr, error_corr = results.intensity(useamp, True)
            write_ff(amp, intensity, error, intensity_corr, error_corr, only)

    ######## MERGE AMPLITUDES REPLACING REGEX MATCHED STRING WITH WHAT COMES AFTER ~> #########
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
                
                if not test_regex:
                    intensity, error = results.intensity(amps, False)
                    intensity_corr, error_corr = results.intensity(amps, True)
                    write_ff(merged_amp, intensity, error, intensity_corr, error_corr, only)
            merged.clear()
    
    ########## CHECK IF results_dump.coherent_sums is in yaml_file ##########
    if isinstance(yaml_file, (dict, DictConfig)) and "result_dump" in yaml_file and "coherent_sums" in yaml_file["result_dump"]:
        coherent_sums = yaml_file["result_dump"]["coherent_sums"]
        print("\nMerging Amplitude Groups based on user request in YAML field `result_dump.coherent_sums`")
        merged = {}
        for k, vs in coherent_sums.items():
            vs = vs.split("_")
            print(f" -> {k} merged {len(vs)} amplitudes:")
            for amp in uniqueAmps:
                if any([v in amp for v in vs]):
                    print(f"     {amp}")
                    if k not in merged:
                        merged[k] = [amp]
                    else:
                        merged[k].append(amp)
            intensity, error = results.intensity(merged[k], False)
            intensity_corr, error_corr = results.intensity(merged[k], True)
            write_ff(k, intensity, error, intensity_corr, error_corr, only)

    ######### WRITE ALL POSSIBLE PHASE DIFFERENCES ##########
    if not no_phases and not test_regex:
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

    if not test_regex and outfileName != "":
        outfile.close()


def _cli_extract_ff():
    """Command line interface for extracting fit fractions from an amptools fit results"""

    ############## PARSE COMMANDLINE ARGUMENTS ##############
    parser = argparse.ArgumentParser(description="Extract Fit Fractions from FitResults")
    parser.add_argument("fitFile", type=str, default="", help="Amptools FitResults file name")
    parser.add_argument("--outputfileName", type=str, default="extracted_fitfracs.txt", help="Output file name")
    parser.add_argument("--fmt", type=str, default=".5f", help="Format string for printing")
    parser.add_argument("--test_regex", action="store_true", help="Only test regex grouping without calculating intensities")
    parser.add_argument("--no_phases", action="store_true", help="Do not dump phase differences")
    parser.add_argument("--only", type=str, default=None, help="Only dump fit fractions for ['acc', 'noacc']")
    parser.add_argument("--regex_merge", type=str, nargs="+", help="Merge amplitudes: Regex pair (pattern, replace) separated by ~>")
    parser.add_argument("--yaml_file", type=str, default="", help="YAML file name")
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
    yaml_file = args.yaml_file
    if yaml_file != "":
        yaml_file = load_yaml(yaml_file)
    assert os.path.isfile(fitFile), "Fit Results file does not exist at specified path"

    ############## LOAD FIT RESULTS OBJECT ##############
    results = FitResults(fitFile)
    if not results.valid():
        print(f"Invalid fit result in file: {fitFile}")
        exit()

    ############## EXTRACT FIT FRACTIONS ##############
    extract_ff(results, outfileName, fmt, args.test_regex, no_phases, only, regex_merge, yaml_file)

if __name__ == "__main__":
    _cli_extract_ff()
