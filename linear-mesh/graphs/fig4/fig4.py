import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np

def svg_to_data_conv(svg, a, b, thresh=0):
    bs = BeautifulSoup(f)
    num = -1
    for en, i in enumerate(bs.find_all("path")):
        if "js-line" in i["class"]:
            num = en

    vals = bs.find_all("path")[num]["d"].split("L")[1:]
    data = []
    unique = []
    xs = []
    for val in vals:
        x = float(val.split(",")[0])/100
        y_val = float(val.split(",")[1])
        if x>thresh:
            if y_val not in unique:
                unique.append(y_val)
            xs.append(x)
            data.append(a*y_val + b)
            
    print(np.max(unique), np.min(unique))
    return xs, data

def generate_station_data(samples, count):
    repeats = len(samples)//count
    arr = np.arange(count).repeat(repeats)
    x = np.arange(len(samples), dtype=np.float32)/len(samples)
    x *= np.max(samples)
    remains = len(samples)-len(arr)

    for i in range(remains):
        index = np.random.randint(len(arr))
        arr = np.insert(arr, index, arr[index])
    
    return x, arr

with open("CW_ddpg.svg") as f:
    a = -2.382996994418204
    b = 1080.005367110347
    x_data, data = svg_to_data_conv(f, a, b)
    x_data = x_data[0:len(x_data):7]
    data = data[0:len(data):7]

with open("CW_dqn.svg") as f:
    a = -2.382996994418204
    b = 1080.005367110347
    x_dqn, dqn = svg_to_data_conv(f, a, b)

x_station, stations = generate_station_data(x_dqn, 50)

fig, ax1 = plt.subplots()
plt.xlabel("Simulation time [s]")

ax1.set_ylabel('CW')
plt.plot(x_dqn, dqn, label="CCOD w/ DQN")
ax1.plot(x_data, data, label="CCOD w/ DDPG", color="#f7b051")
ax1.tick_params(axis='y')
plt.legend()

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Transmitting stations', color=color)
ax2.plot(x_station, stations, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.xticks(np.arange(7)*7/6, np.arange(7)*10)
plt.xlim([0, 7])
plt.show()
