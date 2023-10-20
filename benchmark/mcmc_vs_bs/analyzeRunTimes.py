import glob
import numpy as np
import matplotlib.pyplot as plt
from utils import setPlotStyle

setPlotStyle(small_size=18, big_size=22)

folders = [ 'real_mi_example', 'sdme_example'] #, 'real_md_example' ]
labels  = [ r'$\eta\pi^0$ Mass Indep.', r'$\rho$ SDME' ] #, r'$\eta\pi^0$ Mass Dep.'


print("\n========= MLE ITERATION TIMES =======")
mle_mean_fit_times = []
mle_std_fit_times = []
for folder in folders:
    fit_times = []
    files = glob.glob(f'{folder}_mle/*_mle.out')
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()[-10:]
            for line in lines:
                if "Fit time:" in line:
                    fit_time = float(line.split()[2])
                if "Total time:" in line:
                    total_time = float(line.split()[2])
            time = line.split()
            fit_times.append(fit_time)
    mean, std = np.mean(fit_times), np.std(fit_times)
    print(f'{folder} fit time: {mean:.2f} +/- {std:.2f} seconds')
    mle_mean_fit_times.append(mean)
    mle_std_fit_times.append(std)
mle_mean_fit_times = np.array(mle_mean_fit_times)
mle_std_fit_times = np.array(mle_std_fit_times)
print("=====================================\n")


print("\n========== MCMC SAMPLE TIMES ========")
mcmc_mean_fit_times = []
mcmc_std_fit_times = []
for folder in folders:
    fit_times = []
    files = glob.glob(f'{folder}_mcmc/*_mcmc.out')
    for file in files:
        fit_time, total_time, nSamples = 0, 0, 0
        with open(file, 'r') as f:
            lines = f.readlines()#[-10:]
            for line in lines:
                if "Fit time:" in line:
                    fit_time = float(line.split()[2])
                if "Total time:" in line:
                    total_time = float(line.split()[2])
                if "MAP Estimates" in line:
                    nSamples = int(line.split()[3])
            time = line.split()
            fit_times.append(fit_time)
    mean, std = np.mean(fit_times), np.std(fit_times)
    mean, std = mean/nSamples, std/nSamples
    print(f'{folder} fit time: {mean:.5f} +/- {std:.5f} seconds')
    mcmc_mean_fit_times.append(mean)
    mcmc_std_fit_times.append(std)
mcmc_mean_fit_times = np.array(mcmc_mean_fit_times)
mcmc_std_fit_times = np.array(mcmc_std_fit_times)
print("=====================================\n")

ratio_mean_fit_times = mle_mean_fit_times / mcmc_mean_fit_times
ratio_std_fit_times = ratio_mean_fit_times * np.sqrt( (mle_std_fit_times/mle_mean_fit_times)**2 + (mcmc_std_fit_times/mcmc_mean_fit_times)**2 )

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.errorbar(range(len(folders)), ratio_mean_fit_times, yerr=ratio_std_fit_times, fmt='o', c='black')
plt.xticks(range(len(labels)), labels)
plt.xlim(-0.25, len(labels)-0.75)
plt.ylabel("MCMC / MLE Rate Ratio")
plt.xlabel("Analysis")
plt.savefig('benchmark_mcmc_vs_mle.png')
