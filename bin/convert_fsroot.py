#!/usr/bin/env python3
import ROOT
import argparse
import re
import sys
from array import array

# -----------------------------------------------------------------------------#
# Purpose:
#   Read a ROOT tree where particle kinematics are stored as individual scalar
#   branches of the form P{x,y,z}P{B,1,2,…} and EnP{B,1,2,…}, and convert it
#   into another AmpTools structure:
#     • Beam quantities:
#         – E_Beam, Px_Beam, Py_Beam, Pz_Beam      (all floats)
#     • Final‐state particle arrays (length = NumFinalState):
#         – E_FinalState[NumFinalState]/F
#         – Px_FinalState[NumFinalState]/F
#         – Py_FinalState[NumFinalState]/F
#         – Pz_FinalState[NumFinalState]/F
#     • NumFinalState    (integer, number of final‐state particles)
#     • Weight           (float; all ones if no weight input provided)
#
# Expected inputs:
#   Required positional arguments:
#     input_root     : path to the input ROOT file
#   Optional positional arguments:
#     input_tree     : name of the tree inside input_root (default: first tree in file)
#     output_root    : path for the output ROOT file to be created
#     output_tree    : name of the new tree in output_root (default: 'kin')
#
#   Optional (all-or-none):
#     --weight_root  : path to a ROOT file containing an auxiliary weight tree
#     --weight_tree  : name of the weight tree inside weight_root
#     --weight_branch: name of the branch in weight_tree holding per-entry weights
#     --mc          : use MC-prefixed branches (e.g. MCEnPB, MCPxPB) instead of regular ones
#
# Input‐tree branch requirements:
#   • Beam momentum branches:      PxPB, PyPB, PzPB      (float) [or MCPxPB, MCPyPB, MCPzPB with --mc]
#   • Beam energy branch:          EnPB                 (float) [or MCEnPB with --mc]
#   • Final‐state momentum:        PxP1, PyP1, PzP1 …   (float) [or MCPxP1, MCPyP1, MCPzP1 … with --mc]
#   • Final‐state energy:          EnP1, EnP2, …        (float) [or MCEnP1, MCEnP2, … with --mc]
#     – Particle indices must be numeric and start at 1
#
# Behavior:
#   • If input_tree is not specified, automatically selects the first tree found in the input file.
#   • If output_tree is not specified, defaults to 'kin'.
#   • If --mc flag is used, reads from MC-prefixed branches (e.g. MCEnPB, MCPxPB) instead of regular ones.
#   • Verifies all required branches exist; exits with an error if any are missing.
#   • If weight arguments are provided, checks that the weight tree has the same
#     number of entries as the input tree and uses its values for Weight.
#   • Otherwise, fills Weight = 1.0 for every entry.
#   • Writes out the new tree with renamed and restructured branches.
#
# -----------------------------------------------------------------------------


def convert_tree(input_root_path, input_tree_name,
                 output_root_path, output_tree_name,
                 weight_root_path=None, weight_tree_name=None, weight_branch_name=None, use_mc=False):
    # 1) Open input file & tree
    input_file = ROOT.TFile.Open(input_root_path)
    if not input_file or input_file.IsZombie():
        sys.exit(f"ERROR: cannot open input file '{input_root_path}'")
    
    # If no input tree name provided, use the first tree in the file
    if input_tree_name is None:
        keys = input_file.GetListOfKeys()
        tree_keys = [key.GetName() for key in keys if key.GetClassName() == 'TTree']
        if not tree_keys:
            sys.exit(f"ERROR: no trees found in '{input_root_path}'")
        input_tree_name = tree_keys[0]
        print(f"INFO: Using first tree '{input_tree_name}' from input file")
    
    in_tree = input_file.Get(input_tree_name)
    if not in_tree:
        sys.exit(f"ERROR: tree '{input_tree_name}' not found in '{input_root_path}'")

    # 2) (Optional) Open weight file & tree
    use_weights = False
    if weight_root_path and weight_tree_name and weight_branch_name:
        wfile = ROOT.TFile.Open(weight_root_path)
        if not wfile or wfile.IsZombie():
            sys.exit(f"ERROR: cannot open weight file '{weight_root_path}'")
        wtree = wfile.Get(weight_tree_name)
        if not wtree:
            sys.exit(f"ERROR: tree '{weight_tree_name}' not found in '{weight_root_path}'")
        if not wtree.GetBranch(weight_branch_name):
            sys.exit(f"ERROR: branch '{weight_branch_name}' not found in weight tree")
        if wtree.GetEntries() != in_tree.GetEntries():
            sys.exit("ERROR: input tree and weight tree have different numbers of entries")
        use_weights = True

    # 3) Discover beam vs. final‐state branches
    branch_list = [b.GetName() for b in in_tree.GetListOfBranches()]
    
    # Choose prefix based on MC flag
    prefix = "MC" if use_mc else ""
    mom_patt = re.compile(f'^{prefix}P([xyz])P(B|\\d+)$')      # e.g. PxPB or MCPxPB
    ene_patt = re.compile(f'^{prefix}EnP(B|\\d+)$')            # e.g. EnPB or MCEnPB

    # store beam components
    beam_mom = {}
    final_mom = {}   # pid -> {comp: branch}
    beam_E = None
    final_E = {}     # pid -> branch

    for name in branch_list:
        m = mom_patt.match(name)
        if m:
            comp, pid = m.group(1), m.group(2)
            if pid == 'B':
                beam_mom[comp] = name
            else:
                final_mom.setdefault(pid, {})[comp] = name
        else:
            m2 = ene_patt.match(name)
            if m2:
                pid = m2.group(1)
                if pid == 'B':
                    beam_E = name
                else:
                    final_E[pid] = name

    # checks: beam momentum
    for c in ('x','y','z'):
        if c not in beam_mom:
            sys.exit(f"ERROR: missing input branch {prefix}P{c.upper()}PB for beam momentum")
    # beam energy
    if beam_E is None:
        sys.exit(f"ERROR: missing input branch {prefix}EnPB for beam energy")

    # final‐state indices
    fids = sorted(final_mom.keys(), key=lambda x: int(x))
    n_final = len(fids)
    if n_final == 0:
        example_branches = f"{prefix}PxP1, {prefix}PyP1, …" if prefix else "PxP1, PyP1, …"
        sys.exit(f"ERROR: no final‐state momentum branches found (e.g. {example_branches})")

    # check each final state has all momentum comps + EnP
    for pid in fids:
        comps = final_mom[pid]
        for c in ('x','y','z'):
            if c not in comps:
                sys.exit(f"ERROR: missing input branch {prefix}P{c.upper()}P{pid} for final‐state {pid}")
        if pid not in final_E:
            sys.exit(f"ERROR: missing input branch {prefix}EnP{pid} for final‐state energy")

    # 4) Prepare the output file & tree
    out_file = ROOT.TFile.Open(output_root_path, "RECREATE")
    out_tree = ROOT.TTree(output_tree_name, output_tree_name)

    # beam buffers
    buf_Eb  = array('f',[0.])
    buf_Pxb = array('f',[0.])
    buf_Pyb = array('f',[0.])
    buf_Pzb = array('f',[0.])
    out_tree.Branch('E_Beam',  buf_Eb,  "E_Beam/F")
    out_tree.Branch('Px_Beam', buf_Pxb, "Px_Beam/F")
    out_tree.Branch('Py_Beam', buf_Pyb, "Py_Beam/F")
    out_tree.Branch('Pz_Beam', buf_Pzb, "Pz_Beam/F")

    # NumFinalState and Weight - MUST come before variable-size arrays
    buf_Nfs = array('i',[n_final])
    buf_W   = array('f',[1.])
    out_tree.Branch('NumFinalState', buf_Nfs, "NumFinalState/I")
    out_tree.Branch('Weight',         buf_W,   "Weight/F")

    # final‐state arrays - now NumFinalState branch exists
    buf_Ef  = array('f', [0.]*n_final)
    buf_Pxf = array('f', [0.]*n_final)
    buf_Pyf = array('f', [0.]*n_final)
    buf_Pzf = array('f', [0.]*n_final)
    out_tree.Branch('E_FinalState',  buf_Ef,  "E_FinalState[NumFinalState]/F")
    out_tree.Branch('Px_FinalState', buf_Pxf, "Px_FinalState[NumFinalState]/F")
    out_tree.Branch('Py_FinalState', buf_Pyf, "Py_FinalState[NumFinalState]/F")
    out_tree.Branch('Pz_FinalState', buf_Pzf, "Pz_FinalState[NumFinalState]/F")

    # 5) Link input branches with proper data types
    in_tree.SetBranchStatus('*', 0)
    
    # beam - use double arrays for Double_t branches
    be_arr = {}
    for comp, br in (('E', beam_E),
                     ('x', beam_mom['x']),
                     ('y', beam_mom['y']),
                     ('z', beam_mom['z'])):
        arr = array('d',[0.])  # Use double for Double_t branches
        in_tree.SetBranchStatus(br,1)
        in_tree.SetBranchAddress(br, arr)
        be_arr[comp] = arr

    # final states - use double arrays for Double_t branches
    fs_arr = {}
    for pid in fids:
        for comp in ('E','x','y','z'):
            if comp == 'E':
                br = final_E[pid]
            else:
                br = final_mom[pid][comp]
            
            arr = array('d',[0.])  # Use double for Double_t branches
            in_tree.SetBranchStatus(br,1)
            in_tree.SetBranchAddress(br, arr)
            fs_arr[(pid,comp)] = arr

    # weight buffer - use double array for Double_t branch
    if use_weights:
        w_arr = array('d',[1.])  # Use double for Double_t branch
        wtree.SetBranchStatus(weight_branch_name,1)
        wtree.SetBranchAddress(weight_branch_name, w_arr)

    # 6) Loop
    nent = in_tree.GetEntries()
    for i in range(nent):
        in_tree.GetEntry(i)
        if use_weights:
            wtree.GetEntry(i)
            buf_W[0] = w_arr[0]
        else:
            buf_W[0] = 1.0

        buf_Eb[0]  = be_arr['E'][0]
        buf_Pxb[0] = be_arr['x'][0]
        buf_Pyb[0] = be_arr['y'][0]
        buf_Pzb[0] = be_arr['z'][0]

        buf_Nfs[0] = n_final
        for idx, pid in enumerate(fids):
            buf_Ef[idx]  = fs_arr[(pid,'E')][0]
            buf_Pxf[idx] = fs_arr[(pid,'x')][0]
            buf_Pyf[idx] = fs_arr[(pid,'y')][0]
            buf_Pzf[idx] = fs_arr[(pid,'z')][0]

        out_tree.Fill()

    # 7) Write and close
    out_file.Write()
    out_file.Close()
    input_file.Close()
    if use_weights:
        wfile.Close()

def main():
    p = argparse.ArgumentParser(
        description="Convert P{x,y,z}P* + EnP* → beam+array format with Weight")
    p.add_argument("input_root",  help="input ROOT file")
    p.add_argument("input_tree",  help="input tree name (default: first tree in file)", nargs='?', default=None)
    p.add_argument("output_root", help="output ROOT file")
    p.add_argument("output_tree", help="output tree name (default: 'kin')", nargs='?', default='kin')
    p.add_argument("--weight_root",  help="ROOT file containing weight tree", default=None)
    p.add_argument("--weight_tree",  help="name of weight tree", default=None)
    p.add_argument("--weight_branch",help="branch name holding weights", default=None)
    p.add_argument("--mc", action="store_true", help="use MC-prefixed branches (e.g. MCEnPB, MCPxPB)")
    args = p.parse_args()

    convert_tree(
        args.input_root, args.input_tree,
        args.output_root, args.output_tree,
        args.weight_root, args.weight_tree, args.weight_branch, args.mc
    )

if __name__ == "__main__":
    main() 