import argparse
import uproot as up
import os
import numpy as np

def get_nentries(file_name, delete_if_empty=False, integrate_branch=None):
    with up.open(file_name) as f:
        nkeys = len(f.keys())
        nentries = 0
        integrated_entries = 0

        if nkeys != 0:
            if nkeys > 1:
                print("Warning: more than one tree in file, selecting first")
            t = f[f.keys()[0]]  # get first tree
            nentries = t.num_entries

            if integrate_branch != "None":
                if integrate_branch not in t.keys():
                    return nentries, "Branch not found"
                branch = t[integrate_branch]  # Get the branch first
                integrated_entries = np.sum(branch.array())  # Convert to array and sum

    if nentries == 0 and delete_if_empty:
        print("Found empty file, deleting...")
        os.remove(file_name)

    return nentries, integrated_entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get number of entries in all root files")
    parser.add_argument("file_locations", nargs="+", default="*.root", help="File locations (* wildcard support) to get number of entries from")
    parser.add_argument("-i", "--integrate_branch", default='None', help="Provide the integral over this branch") # Useful to determine yield (i.e. summing over Weight branch)
    parser.add_argument("--clean", action="store_true", help="Delete file if empty")

    args = parser.parse_args()
    file_locations = args.file_locations
    clean = args.clean
    integrate_branch = args.integrate_branch
    
    files = sorted(file_locations)

    output = ""
    for file in files:
        entries, integrated_entries = get_nentries(file, delete_if_empty=clean, integrate_branch=integrate_branch)
        output += f"{os.path.basename(file):<20}: Total Entries: {entries}"
        if integrate_branch != "None":
            output += f" | Integrated Entries (branch: {integrate_branch}): {integrated_entries}"
        output += "\n"

    print(output)
