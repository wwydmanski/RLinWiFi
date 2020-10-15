#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import comet_ml.api
import scipy.stats
plt.rcParams.update({'font.size': 14})

# Avoid Type 3 fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import cycler
n = 2 #number of lines
color = plt.cm.Blues(np.linspace(0.5, 0.8,n)) #gnuplot - Blues name, linspace parameters determine the boundaries of the color
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
#%%
comet_api = comet_ml.api.API("haQ9dJrZ4oZHhZhX8O7JJ2AJ5")

def extract_values(experiment_key):
    res = comet_api.get(f"wwydmanski/rl-in-wifi/{experiment_key}")
    res = res.get_metrics("Chosen CW")
    values = np.array([float(i['metricValue']) for i in res], dtype=np.float)
    steps = np.array([i['step'] for i in res], dtype=np.float)
    
    round_ends = [0]
    round_ends.extend([np.where(steps<i*6300)[0][-1] for i in range(1, 16)])
    means = []
    for i in range(len(round_ends)):
        means.append(np.mean(values[round_ends[i-1]:round_ends[i]]))
    return means[1:]
#%%

def get_metrics(experiments):
    res = np.array([extract_values(i) for i in experiments])
    for exp in res:
        for i in range(len(exp)):
            if np.isnan(exp[i]):
                exp[i] = (exp[i-1]+exp[i+1])/2
    mean = np.mean(res, axis=0)
    std = np.std(res, axis=0)
    alpha = 0.05
    n = len(experiments)
    yerr = std / np.sqrt(n) * scipy.stats.t.ppf(1-alpha/2, n - 1)
    upper = yerr.copy()
    lower = yerr.copy()
    for i in range(len(yerr)):
        if mean[i]-lower[i]<0:
            lower[i] = mean[i]
    intervals = (lower, upper)
    return mean, intervals

plt.figure(figsize=(6.4, 4.8),dpi=100)

means, yerr = get_metrics(["918819ef9ae14477b2cf0866e35af8f8", "ec240e737d3e472b819c51d49a4d97bf"])
plt.errorbar(np.arange(len(means))+1, means, yerr, fmt='.-', label="CCOD w/ DQN", marker="v", markersize=6, capsize=2)

means, yerr = get_metrics(["1043a164b186427f9d17b7b45eeb216c", "4f06b6a983764fe2b30cb5f94241d084", "55ee5bb7ca824c96a8336c8192e6fcea", "f4c1d0bfb2f94657bf3128a7c67c495d"])
plt.errorbar(np.arange(len(means))+1, means, yerr=yerr, fmt='.-', label="CCOD w/ DDPG", marker="^", markersize=6, capsize=2)

plt.xlabel("Round")
plt.ylabel("Mean CW [slots]")
plt.legend(frameon=False)
plt.xticks(np.arange(1, 16, 1.0))
plt.tight_layout()
plt.savefig('cw_vs_rounds.pdf')
# plt.ylim([0, 600])
plt.show()