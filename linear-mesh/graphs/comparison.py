import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
plt.style.use("default")

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'errorbar.capsize': 2})

# Avoid Type 3 fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import cycler
COEFF = 1.048576

n = 4 #number of lines
color = plt.cm.Blues(np.linspace(0.3, 1,n)) #gnuplot - colormap name, 0 and 1 determine the boundaries of the color
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

def _get_yerr(df):
    alpha=0.05
    std = df.groupby('count').std()
    n = df.groupby('count').count()
    #Calculate confidence intervals 
    yerr = std / np.sqrt(n) * st.t.ppf(1-alpha/2, n - 1)
    yerr = yerr.fillna(0)
    return yerr

def plot(scenario):
    if scenario == 'basic':
        BEB = [39.15, 34.47, 30.64, 27.1]
        RANGE = [5, 15, 30, 50]

    elif scenario == 'convergence':
        BEB = [39.94, 37.64, 35.19, 32.73]
        RANGE = [6, 15, 30, 50]

    df = pd.read_csv("final.csv")
    df = df[df['type']==scenario]
    DDPG = df[df['algorithm']=='ddpg'].groupby('count').mean()*COEFF
    DQN = df[df['algorithm']=='dqn'].groupby('count').mean()*COEFF
    STATIC = df[df['algorithm']=='static'].groupby('count').mean()*COEFF

    DDPG_yerr = _get_yerr(df[df['algorithm']=='ddpg'])
    DQN_yerr = _get_yerr(df[df['algorithm']=='dqn'])

    plt.figure(figsize=(6.4, 4.8),dpi=100)

    plt.errorbar(RANGE, BEB, fmt='.-', label="Standard 802.11", marker="s", markersize=6, yerr=[0, 0, 0, 0])
    plt.errorbar(RANGE, STATIC.values, fmt='.-', label="Look-up table", markersize=10, yerr=[0, 0, 0, 0])
    DQN.plot(fmt='.-', label="CCOD w/ DQN", marker="v",markersize=6, yerr=DQN_yerr, ax=plt.gca())
    DDPG.plot(fmt='.-', label='CCOD w/ DDPG', marker="^",markersize=6, yerr=DDPG_yerr, ax=plt.gca())

    # plt.plot(RANGE[:len(DQN)], DQN, '.-', label="CCOD w/ DQN", marker="v",markersize=6)
    # plt.plot(RANGE, DDPG, '.-', label="CCOD w/ DDPG", marker="^",markersize=6)

    plt.xlabel("Number of stations")
    if (scenario=='convergence'):
        plt.xlabel("Final number of stations")
    plt.ylabel("Network throughput [Mb/s]")
    plt.ylim([26, 42])
    plt.xlim([0, 55])
    # plt.title("CONVERGENCE scenario comparison")
    plt.legend(["Standard 802.11", "Look-up table", "CCOD w/ DQN", "CCOD w/ DDPG"], loc=3, frameon=False)
    plt.tight_layout()
    plt.savefig(scenario+'.pdf');
    plt.show()

plot('basic')
plot('convergence')
