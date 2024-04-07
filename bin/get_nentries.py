import argparse
import uproot as up
import os


def get_nentries(file_name, delete_if_empty=False):
    with up.open(file_name) as f:
        nkeys = len(f.keys())
        nentries = 0

        if nkeys != 0:
            if nkeys > 1:
                print("Warning: more than one tree in file, selecting first")
            t = f[f.keys()[0]]  # get first tree
            nentries = t.num_entries

    if nentries == 0 and delete_if_empty:
        print("Found empty file, deleting...")
        os.remove(file_name)

    return nentries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get number of entries in all root files")
    parser.add_argument("file_locations", nargs="+", default="*.root", help="File locations (* wildcard support) to get number of entries from")
    parser.add_argument("--clean", action="store_true", help="Delete file if empty")

    args = parser.parse_args()
    file_locations = args.file_locations
    clean = args.clean

    files = sorted(file_locations)

    total_entries = 0
    for file in files:
        entries = get_nentries(file, delete_if_empty=clean)
        print(f"{os.path.basename(file)}: {entries}")
        total_entries += entries

    print(f"\nTotal Entries: {total_entries}")
