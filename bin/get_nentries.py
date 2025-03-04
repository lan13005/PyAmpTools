import argparse
import uproot as up
import os
import numpy as np
from rich.console import Console

def get_nentries(file_name, delete_if_empty=False, integrate_branch=None, filters=None):
    
    def check_branch_exists(t, branch_name):
        if branch_name not in t.keys():
            raise ValueError(f"Branch {branch_name} not found in file {file_name}")
    
    with up.open(file_name) as f:
        nkeys = len(f.keys())
        nentries = 0
        integrated_entries = 0

        if nkeys != 0:
            if nkeys > 1:
                print("Warning: more than one tree in file, selecting first")
            t = f[f.keys()[0]]  # get first tree
            nentries = t.num_entries
            
            # Apply filters to the tree keeping access to all original branches
            if filters:
                cut_expressions = []
                for branch, min_val, max_val, is_cut in filters:
                    if is_cut:
                        cut_expressions.append(f"({branch} < {min_val}) | ({branch} > {max_val})")
                    else: # is a selection
                        cut_expressions.append(f"({branch} > {min_val}) & ({branch} < {max_val})")
                
                if cut_expressions:
                    full_cut = " & ".join(cut_expressions)
                    filtered_arrays = t.arrays(cut=full_cut)
                    nentries = len(filtered_arrays)
                    if integrate_branch != "None":
                        check_branch_exists(t, integrate_branch)
                        integrated_entries = np.sum(filtered_arrays[integrate_branch])
                    integrate_branch = "None"

            # This should not run if filters have been run
            if integrate_branch != "None":
                check_branch_exists(t, integrate_branch)
                branch_array = t[integrate_branch].array()
                integrated_entries = np.sum(branch_array)

    if nentries == 0 and delete_if_empty:
        print("Found empty file, deleting...")
        os.remove(file_name)

    return nentries, integrated_entries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get number of entries in all root files")
    parser.add_argument("file_locations", nargs="+", default="*.root", help="File locations (* wildcard support) to get number of entries from")
    parser.add_argument("-i", "--integrate_branch", default='None', help="Provide the integral over this branch") # Useful to determine yield (i.e. summing over Weight branch)
    parser.add_argument("-s", "--selections", nargs="+", default=[], help="Provide the selections to apply to the branches. Default is to select / prepend with ! for cuts. Example: branchName1 min1 max1 !branchName2 min2 max2")
    parser.add_argument("--clean", action="store_true", help="Delete file if empty")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()
    file_locations = args.file_locations
    clean = args.clean
    integrate_branch = args.integrate_branch
    selections = args.selections

    if isinstance(selections, str):
        selections = selections.split(" ")
    if not isinstance(selections, list):
        raise ValueError("Selections must be provided as a list or a space delimited string!")
    
    console = Console()
    
    # Parse selections
    filters = []
    if len(selections) > 0:
        if len(selections) % 3 != 0:
            raise ValueError("Selections must be provided in groups of 3 (branch, min, max)")
        for i in range(0, len(selections), 3):
            branch = selections[i]
            min_val = float(selections[i+1])
            max_val = float(selections[i+2])
            is_cut = False
            if branch[0] == "!": # defines a cut
                is_cut = True
                branch = branch[1:] 
            filters.append((branch, min_val, max_val, is_cut))
                
    if args.verbose:
        console.print(f"Processing {len(file_locations)} ROOT file(s)...")
        for f in file_locations:
            console.print(f"Processing file: {f}")
        if len(filters) > 0:
            console.print(f"Applying selections: {filters}")
        console.print(f"Calculating integral over branch: {integrate_branch}")
        if clean:
            console.print(f"Cleaning empty files: {clean}...")
    
    files = sorted(file_locations)

    list_entries = []
    list_integrated_entries = []
    output = ""
    if args.verbose:
        console.print('\n\n****** RESULTS ******')
    for file in files:
        entries, integrated_entries = get_nentries(file, delete_if_empty=clean, integrate_branch=integrate_branch, filters=filters)
        list_entries.append(entries)
        list_integrated_entries.append(integrated_entries)
        if args.verbose:
            output += f"{os.path.basename(file):<20}: Total Entries: {entries}"
            if integrate_branch != "None":
                output += f" | Integrated Entries (branch: {integrate_branch}): {integrated_entries:0.2f}"
            output += "\n"
            console.print(output)
        else:
            console.print(f"file, entries, integral")
            for file, entries, integrated_entries in zip(files, list_entries, list_integrated_entries):
                console.print(f"{file}, {entries}, {integrated_entries}")
