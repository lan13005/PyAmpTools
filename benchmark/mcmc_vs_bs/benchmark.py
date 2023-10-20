import os
import subprocess
import re

REPO_HOME = os.environ['REPO_HOME']

baseCmd = f"python {REPO_HOME}/EXAMPLES/python"
baseDir = f"{REPO_HOME}/tests/samples/"
cfgs = ['REAL_MI_EXAMPLE/fit', 'SDME_EXAMPLE/sdme', 'REAL_MD_EXAMPLE/etapi_result_src_samePhaseD']
mcmc_extra_flags = '--overwrite'
fit_extra_flags  = '--seedfile mle'

def move_files(dest_folder, iBS):
    os.system(f'mkdir -p {dest_folder}')
    os.system(f'mv -f {dest_folder}.err {dest_folder}/{iBS}_{dest_folder}.err 2> /dev/null')
    os.system(f'mv -f {dest_folder}.out {dest_folder}/{iBS}_{dest_folder}.out 2> /dev/null')
    os.system(f'mv -f result_0.fit {dest_folder}/{iBS}_result.fit 2> /dev/null')
    os.system(f'mv -f mle_0.txt  {dest_folder}/{iBS}_mle.txt 2> /dev/null')

cfgs = cfgs[2:3]

################## MLE BOOTSTRAP ##################
def copyAndSeedCfg(src_cfg, seed):
    dest_cfg = src_cfg.split('/')[-1].replace('.cfg', '_seed.cfg')
    with open(src_cfg, 'r') as f:
        lines = f.readlines()
    with open(dest_cfg, 'w') as f:
        for line in lines:
            if "ROOTDataReaderBootstrap" in line and "accmc" not in line:
                line = line.rstrip() + f' {seed}\n'
            f.write(line)
    return dest_cfg

def clean_attempt():
    os.system(f'rm {dest_folder}.err {dest_folder}.out result_0.fit mle_0.txt')

nBS = 200 # bootstrap iterations
for cfg in cfgs:
    nPassed = 0
    for iBS in range(5*nBS):
        dest_folder = cfg.split("/")[0].lower()+"_mle"
        seeded_cfg = copyAndSeedCfg(f'{baseDir}{cfg}_bootstrap.cfg', str(iBS))
        fit_cmd = f'{baseCmd}/fit.py {seeded_cfg} {fit_extra_flags} > {dest_folder}.out 2> {dest_folder}.err'
        try:
            subprocess.call(fit_cmd, shell=True)
        except:
            clean_attempt()
            continue

        if not os.path.exists("result_0.fit"):
            clean_attempt()
            continue

        with open("result_0.fit") as f:
            regex = re.compile(r"lastMinuitCommandStatus\s+0")
            if regex.search( f.read() ):
                print(f"Fit {iBS} passed...")
                move_files(dest_folder, f'{iBS}')
                nPassed += 1
                if nPassed == nBS:
                    break
            else:
                clean_attempt()
            os.system(f'rm {seeded_cfg}')

################# MCMC SAMPLING ##################
for cfg in cfgs:
    bsfolder = cfg.split("/")[0].lower()+"_mle"
    dest_folder = cfg.split("/")[0].lower()+"_mcmc"
    mcmc_cmd = f'python benchmark_mcmc.py {baseDir}{cfg}.cfg {mcmc_extra_flags} --bsfolder {bsfolder} > {dest_folder}.out 2> {dest_folder}.err'
    print(mcmc_cmd)
    os.system(mcmc_cmd)
    move_files(dest_folder, 'mcmc')
    os.system(f'mv -f mcmc {dest_folder} 2> /dev/null')
