import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("default")

plt.rcParams.update({'font.size': 14})

import cycler
n = 4 #number of lines
color = plt.cm.Blues(np.linspace(0.3, 1,n)) #gnuplot - colormap name, 0 and 1 determine the boundaries of the color
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

def plot(scenario):
    if scenario == 'basic':
      BEB = [39.15, 34.47, 30.64, 27.1]
    #   DDPG = [40.83, 39.91, 39.75, 39.65]
    #   DQN = [40.72, 39.7, 39.58, 39.15]
      STATIC = [40.79, 39.9, 39.55, 38.95]
    elif scenario == 'convergence':
      BEB = [39.94, 37.64, 35.19, 32.73]
    #   DDPG = [40.82, 40, 39.72, 39.34]
    #   DQN = [40.69, 39.74, 39.25, 39.11]
      STATIC = [40.59, 39.45, 39.43, 39.32]

    df = pd.read_csv("final.csv")
    df = df[df['type']==scenario]
    DDPG = df[df['algorithm']=='ddpg'].groupby('count').mean()
    DQN = df[df['algorithm']=='dqn'].groupby('count').mean()
    RANGE = [5, 15, 30, 50]
    
    plt.figure(figsize=(6.4, 4.8))
    
    plt.plot(RANGE, BEB, '.-', label="Standard 802.11", marker="s", markersize=6)
    plt.plot(RANGE, STATIC, '.-', label="Look-up table",markersize=10)
    plt.plot(RANGE[:len(DQN)], DQN, '.-', label="CCOD w/ DQN", marker="v",markersize=6)
    plt.plot(RANGE, DDPG, '.-', label="CCOD w/ DDPG", marker="^",markersize=6)
    
    plt.xlabel("Number of stations")
    plt.ylabel("Aggregate network throughput [Mb/s]")
    plt.ylim([26, 42])
    
    # plt.title("CONVERGENCE scenario comparison")
    plt.legend(loc=3)
    plt.savefig(scenario+'.pdf', bbox_inches='tight');
    plt.show()

plot('basic')
plot('convergence')
