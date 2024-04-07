import os
from pyamptools.utility.general import ConfigLoader
from pyamptools.utility.website import check_amptools_complete
from omegaconf import OmegaConf
import argparse
import subprocess

cmd = """#!/usr/bin/bash -l

# The -l is for login shell and is required to get the full environment with modules

#SBATCH --job-name={4}
#SBATCH --output=slurm/output_ift%j.log
#SBATCH --error=slurm/err_ift%j.log

#SBATCH --nodes=1
#SBATCH --ntasks-per-node={0}
#SBATCH --cpus-per-task={1}
#SBATCH --mem-per-cpu={3}
{5}

#SBATCH --time=24:00:00

echo "Activating Conda Environment"
source /w/halld-scshelf2101/lng/Mambaforge/etc/profile.d/conda.sh # conda init

conda activate pyamptools
source {7}/set_environment.sh

{6}

echo "Running..."

# Run the commands below
{2}

"""


def get_jobid(job_output, job_name):
    if job_output == "":
        raise ValueError(f"{job_name} job submission failed! Could be due to insufficient resources?")
    else:
        job_id = job_output.split()[-1]
    return job_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide data into mass bins")
    parser.add_argument("yaml_name", type=str, default="conf/configuration.yaml", help="Path a configuration yaml file")
    args = parser.parse_args()
    yaml_name = args.yaml_name

    yaml_file = OmegaConf.load(yaml_name)

    cfg = ConfigLoader(yaml_file)

    print("\n\n>>>>>>>>>>>>> ConfigLoader >>>>>>>>>>>>>>>")
    base_directory = cfg("base_directory")
    n_processes = n_mass_bins = cfg("n_mass_bins")
    mem_per_cpu = cfg("batch.slurm_mem_per_cpu", 2000)
    n_randomizations = cfg("amptools.n_randomizations", 1)
    n_systematic_variations = cfg("nifty.n_systematic_variations", 0)
    accelerator = cfg("accelerator")
    print("<<<<<<<<<<<<<< ConfigLoader <<<<<<<<<<<<<<\n\n")

    conda_prefix = os.environ["CONDA_PREFIX"]
    activate_d = os.path.join(conda_prefix, "etc/conda/activate.d")

    ## Determine resources to use
    job_name_prefix = "_".join(base_directory.rstrip("/").split("/")[-2:])
    resource = accelerator.split(":")[0]
    if resource == "gpu":
        print(f"SUBMIT| Submitting jobs using {accelerator}")
        slurm_accelerator = f"#SBATCH --partition=gpu\n#SBATCH --gres={accelerator}"
        mps = "nvidia-cuda-mps-control -d"
    elif resource == "cpu" or resource == "":
        print("SUBMIT| Submitting jobs using cpu")
        slurm_accelerator = "\n"
        mps = ""
    else:
        raise ValueError(f"Unknown resource type: {resource}")

    if not os.path.exists("slurm"):
        os.system("mkdir -p slurm")

    # If AmpToolsFits folder contains the expected number of fits then
    # we can skip the AmpTools fitting step and go directly to NIFTY

    ######################################################
    # Writing and submitting the first job (amptools)
    ######################################################

    print("\n\n***************************************************")

    bSkipAmpToolsFit = check_amptools_complete(n_mass_bins, n_randomizations, base_directory)
    if not bSkipAmpToolsFit:
        if os.path.exists("AmpToolsFits"):
            print("AmpToolsFits folder already exists but was incomplete. Deleting it to start fresh.")
            os.system("rm -rf AmpToolsFits")
        AMPTOOLS_CMD = cmd.format(n_processes, 1, f"pa run_divideData {yaml_name}; pa run_mle {yaml_name}", mem_per_cpu, f"mle_{job_name_prefix}", slurm_accelerator, mps, activate_d)
        with open("submit_amptools.sh", "w") as f:
            f.write(AMPTOOLS_CMD)
        result = subprocess.run(["sbatch", "submit_amptools.sh"], capture_output=True, text=True)
        amptools_job = result.stdout
        amptools_job_id = get_jobid(amptools_job, "AmpTools")
        print(f"Running AmpTools fit with job ID: {amptools_job_id}")
    else:
        print("AmpTools fits already exist and is complete. Skipping AmpTools fits.")

    # ######################################################
    # # Writing and submitting the second job (nifty) with a
    # # dependency on the successful completion of the first job
    # ######################################################

    # NIFTY_CMD = cmd.format(1, n_processes, f'pa run_ift {yaml_name}', mem_per_cpu, f'ift_{job_name_prefix}', slurm_accelerator, mps, activate_d)
    # with open('submit_nifty.sh', 'w') as f:
    #     f.write(NIFTY_CMD)
    # _cmd = ['sbatch']
    # if not bSkipAmpToolsFit:
    #     _cmd += [f'--dependency=afterok:{amptools_job_id}']
    # _cmd += ['submit_nifty.sh']
    # result = subprocess.run(_cmd, capture_output=True, text=True)
    # nifty_job_id = result.stdout
    # nifty_job_id = get_jobid(nifty_job_id, 'NIFTy')
    # print(f'Running NIFTy initial fit with job ID: {nifty_job_id}')

    # # ######################################################
    # # # Writing and submitting the third job (systematics)
    # # ######################################################

    # if n_systematic_variations > 0:
    #     print(f'Running {n_systematic_variations} systematic variations')
    #     # Writing and submitting the third job (systematics) with a dependency on the successful completion of the second job
    #     SYSTEMATICS_CMD = cmd.format(1, n_processes, f'pa run_iftsyst {yaml_name}', mem_per_cpu, f'ift_{job_name_prefix}_syst', slurm_accelerator, mps, activate_d)
    #     with open('submit_systematics.sh', 'w') as f:
    #         f.write(SYSTEMATICS_CMD)
    #     _cmd = f'sbatch --dependency=afterok:{nifty_job_id} submit_systematics.sh'
    #     _cmd = _cmd.split()
    #     result = subprocess.run(_cmd, capture_output=True, text=True)
    #     systematics_job = result.stdout
    #     systematics_job_id = get_jobid(systematics_job, 'Systematics')
    #     print(f'Running Systematics with job ID: {systematics_job_id}')
    # else:
    #     print('No systematic variations requested')

    print("***************************************************\n\n")
