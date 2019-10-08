import matplotlib.pyplot as plt

# plt.style.use("seaborn")

BEB = [38.39, 35.3, 33.6, 29.75]
DDPG = [40.82, 40, 39.72, 39.34]
DQN = [40.69, 39.74, 39.25, 39.11]

RANGE = [5, 15, 30, 50]

plt.plot(RANGE, BEB, 'o-', label="BEB")
plt.plot(RANGE, DDPG, 'o-', label="DDPG")
plt.xlabel("Number of stations")
plt.ylabel("Speed [Mb/s]")
plt.plot(RANGE[:len(DQN)], DQN, 'o-', label="DQN")
plt.title("CONVERGENCE scenario comparison")
plt.legend(loc=3)
plt.show()