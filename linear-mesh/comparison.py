import matplotlib.pyplot as plt

plt.style.use("seaborn")

BEB = [39.15, 34.47, 30.64, 27.1]
DDPG = [40.83, 39.91, 39.64, 39.65]
DQN = [39.66, 38.79]

RANGE = [5, 15, 30, 50]

plt.plot(RANGE, BEB, 'o-', label="BEB")
plt.plot(RANGE, DDPG, 'o-', label="DDPG")
plt.xlabel("Number of stations")
plt.ylabel("Speed [Mb/s]")
plt.plot(RANGE[:2], DQN, 'o-', label="DQN")
plt.title("BASIC scenario comparison")
plt.legend()
plt.show()