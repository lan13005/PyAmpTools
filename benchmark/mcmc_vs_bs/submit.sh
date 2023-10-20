#!/usr/bin/bash -l

# The -l is for login shell and is required to get the full environment with modules

#SBATCH --job-name=PyAmpTools
#SBATCH --output=slurm/output.log
#SBATCH --error=slurm/err.log

#SBATCH --nodes=1                     # Needs to match --num_nodes passed to Trainer()
#SBATCH --ntasks-per-node=2           # PER NODE: Needs to match --devices passed to Trainer(), which should be your choice in gres
#SBATCH --gres=gpu:A100:1               # PER NODE: ifarm nodes can have up to 4 TitanRTX or 4 A100 or (8/16) T4 GPUs
#SBATCH --cpus-per-task=1             # PER TASK: Number of CPU cores per task, I think it could be used by DataLoader(num_workers) flag
#SBATCH --mem=12G                     # PER NODE: Requested memory for each node
#SBATCH --partition=gpu
#SBATCH --time=36:00:00

echo "Activating Conda Environment"
source /w/halld-scshelf2101/lng/Mambaforge/etc/profile.d/conda.sh # conda init
conda activate PyAmpTools
echo "Running..."

cd /w/halld-scshelf2101/lng/WORK/PyAmpTools/benchmark/mcmc_vs_bs
python benchmark.py
