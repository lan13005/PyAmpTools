from pyamptools.utility.general import append_kinematics
from rich.console import Console
import numpy as np
import os
import argparse

console = Console()

def split_mass_t(
    infile, 
    outputBase, 
    lowMass, 
    highMass, 
    nMBins, 
    lowT, 
    highT, 
    nTBins,
    treeName="kin",
    mass_edges=None, 
    t_edges=None,
    dump_augmented_tree=False,
    overwrite=False,
):
    """
    Split events into bins based on (invariant mass, t) and save to separate ROOT files.
    
    Args:
        infile (str): Input ROOT file
        outputBase (str): Base name for output ROOT files
        lowMass (float): Lower bound of mass range
        highMass (float): Upper bound of mass range
        nMBins (int): Number of mass bins
        lowT (float): Lower bound of t range
        highT (float): Upper bound of t range
        nTBins (int): Number of t bins
        treeName (str): Name of the TTree in the input ROOT file
        mass_edges (List[float]): Bin the data with these bin edges (nMBins + 1 elements), default None
        t_edges (List[float]): Bin the data with these bin edges (nTBins + 1 elements), default None
        dump_augmented_tree (bool): Whether to save the augmented tree with all events
        overwrite (bool): If True, force recalculation of derived kinematics even if they already exist
    
    Returns:
        tuple: (mass_edges, t_edges, nBar, nBar_err)
    """
    
    console = Console()
    
    # Generate bin edges if not provided
    if mass_edges is None:
        mass_edges = np.linspace(lowMass, highMass, nMBins + 1)
    else:
        mass_edges = np.array(mass_edges)
    
    if t_edges is None:
        t_edges = np.linspace(lowT, highT, nTBins + 1)
    else:
        t_edges = np.array(t_edges)

    console.print(f"Appending kinematics to {infile}", style="bold blue")
    df, kin_quantities = append_kinematics(infile, None, treeName, console=console, overwrite=overwrite)

    # Initialize arrays to store the bin information
    nBar = np.zeros((nMBins, nTBins))
    nBar_err = np.zeros((nMBins, nTBins))
    
    # Create separate dataframes for each (mMassX, t) bin
    for i in range(nMBins):
        mass_low = mass_edges[i]
        mass_high = mass_edges[i+1]
        
        for j in range(nTBins):
            t_low = t_edges[j]
            t_high = t_edges[j+1]
            
            # Calculate bin index as in split_mass_t function
            k = j * nMBins + i
            
            bin_output_root = f"{outputBase}_{k}.root"
            fname = os.path.basename(bin_output_root)
            if os.path.exists(bin_output_root):
                console.print(f"File {fname} already exists, skipping bin {k}", style="bold yellow")
                continue
            
            bin_df = df.Filter(f"mMassX >= {mass_low} && mMassX < {mass_high} && mt >= {t_low} && mt < {t_high}")
            
            nentries = bin_df.Count().GetValue()

            if "Weight" in df.GetColumnNames():
                integral = bin_df.Sum("Weight").GetValue()
                # Calculate error as sqrt(sum of squared weights)
                sq_weights = bin_df.Define("WeightSquared", "Weight*Weight")
                error = np.sqrt(sq_weights.Sum("WeightSquared").GetValue())
            else:  # unweighted
                integral = nentries
                error = np.sqrt(integral)
            
            # Store values for return
            nBar[i, j] = integral
            nBar_err[i, j] = error
            
            # Raise error if any empty bins are found
            if nentries == 0:
                console.print(f"ERROR: No events in bin (massBin={i}, tBin={j}). Currently we cannot proceed with empty bins. Please modify your mass and t ranges and binning!", style="bold red")
                exit()
            
            bin_df.Snapshot(treeName, bin_output_root)
            console.print(f"Writing {nentries} events ({integral:0.2f} weighted events) to {outputBase}_{k}.root ~> (massBin={i}, tBin={j})...", style="bold green")
    
    # Round the edges
    mass_edges = [float(np.round(edge, 5)) for edge in mass_edges]
    t_edges = [float(np.round(edge, 5)) for edge in t_edges]

    return mass_edges, t_edges, nBar, nBar_err

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split events into bins based on invariant mass and save to separate ROOT files.")
    parser.add_argument("infile", type=str, help="Input ROOT file")
    parser.add_argument("outputBase", type=str, help="Base name for output ROOT files, i.e. ./myfile will dump to ./myfile_0.root, ./myfile_1.root, etc.")
    parser.add_argument("lowMass", type=float, help="Lower bound of mass range")
    parser.add_argument("highMass", type=float, help="Upper bound of mass range")
    parser.add_argument("nBins", type=int, help="Number of mass bins")
    parser.add_argument("lowT", type=float, help="Lower bound of t range")
    parser.add_argument("highT", type=float, help="Upper bound of t range")
    parser.add_argument("nTBins", type=int, help="Number of t bins")
    parser.add_argument("--treeName", type=str, default="kin", help="Name of the TTree in the input ROOT file")
    parser.add_argument("--mass_edges", type=list, default=None, help="Bin the data with these mass-bin edges (nBins + 1 elements). If None, will be computed based on other cfg options")
    parser.add_argument("--t_edges", type=list, default=None, help="Bin the data with these t-bin edges (nBins + 1 elements). If None, will be computed based on other cfg options")
    parser.add_argument("--overwrite", action="store_true", help="Force recalculation of kinematics even if they already exist")
    args = parser.parse_args()

    split_mass_t(args.infile, args.outputBase, 
                args.lowMass, args.highMass, args.nBins, 
                args.lowT, args.highT, args.nTBins,
                args.treeName, args.mass_edges, args.t_edges, args.overwrite)
