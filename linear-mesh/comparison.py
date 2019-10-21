import matplotlib.pyplot as plt

# plt.style.use("seaborn")

## BASIC scenario
# BEB = [39.15, 34.47, 30.64, 27.1]
# DDPG = [40.83, 39.91, 39.75, 39.65]
# DQN = [40.72, 39.7, 39.58, 39.15]
# STATIC = [40.79, 39.9, 39.55, 38.95]

## CONVERGENCE scenario
BEB = [38.39, 35.3, 33.6, 29.75]
DDPG = [40.82, 40, 39.72, 39.34]
DQN = [40.69, 39.74, 39.25, 39.11]
STATIC = [40.59, 39.45, 39.43, 39.32]

RANGE = [5, 15, 30, 50]

plt.plot(RANGE, BEB, '.-', label="BEB")
plt.plot(RANGE, DDPG, '.-', label="DDPG")
plt.plot(RANGE, STATIC, '.-', label="STATIC")
plt.xlabel("Number of stations")
plt.ylabel("Speed [Mb/s]")
plt.plot(RANGE[:len(DQN)], DQN, '.-', label="DQN")
# plt.title("CONVERGENCE scenario comparison")
plt.legend(loc=3)
plt.show()