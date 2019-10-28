import matplotlib.pyplot as plt
import numpy as np
#plt.style.use("seaborn-white")
plt.style.use("default")

def plot(scenario):
    if scenario == 'static':
      BEB = [39.15, 34.47, 30.64, 27.1]
      DDPG = [40.83, 39.91, 39.75, 39.65]
      DQN = [40.72, 39.7, 39.58, 39.15]
      STATIC = [40.79, 39.9, 39.55, 38.95]
    elif scenario == 'dynamic':
      BEB = [39.94, 37.64, 35.19, 32.73]
      DDPG = [40.82, 40, 39.72, 39.34]
      DQN = [40.69, 39.74, 39.25, 39.11]
      STATIC = [40.59, 39.45, 39.43, 39.32]
    
    RANGE = [5, 15, 30, 50]
    
    plt.plot(RANGE, DDPG, '.-', label="CCOD w/ DDPG", color="#f7b051", marker="^",markersize=8)
    plt.plot(RANGE[:len(DQN)], DQN, '.-', label="CCOD w/ DQN", color="#e53f26", marker="v",markersize=8)
    plt.plot(RANGE, STATIC, '.-', label="Look-up table (CW=$2^x$)", color="#8a1c1b",markersize=12)
    plt.plot(RANGE, BEB, '.-', label="Standard 802.11", color="#000000", marker="s", markersize=6)
    
    plt.xlabel("Number of stations")
    plt.ylabel("Aggregate network throughput [Mb/s]")
    
    # plt.title("CONVERGENCE scenario comparison")
    plt.legend(loc=3)
    plt.savefig(scenario+'.pdf', bbox_inches='tight');
    plt.show()

plot('static')
plot('dynamic')


## Barplots
# BEB = np.array(BEB)
# DDPG = np.array(DDPG)/BEB - 1
# DQN = np.array(DQN)/BEB - 1
# STATIC = np.array(STATIC)/BEB - 1

# labels = ['G1', 'G2', 'G3', 'G4']
# men_means = [20, 34, 30, 35]
# women_means = [25, 32, 34, 20]

# x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars
# space = 0.03

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width - space, DQN*100, width, label='DQN')
# rects2 = ax.bar(x, STATIC*100, width, label='Static')
# rects3 = ax.bar(x + width + space, DDPG*100, width, label='DDPG')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Improvement [%]')
# ax.set_title('Improvement over BEB')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend(loc=2)


# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height:.1f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)

# fig.tight_layout()
# plt.ylim([0, 40])
# plt.show()