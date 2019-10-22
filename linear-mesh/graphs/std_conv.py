import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np

def svg_to_data_conv(svg, a, b, thresh=0):
    bs = BeautifulSoup(f)
    vals = bs.find_all("path")[-4]["d"].split("L")[1:]
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

# with open("Chosen CW.svg") as f:
#     a = -1.191
#     b = 539.949
#     x_data, data = svg_to_data_conv(f, a, b)

with open("CT_beb.svg") as f:
    a = -0.04685579196217462
    b = 47.5011111111111
    x_beb, beb = svg_to_data_conv(f, a, b)

with open("CT_dqn.svg") as f:
    a = -0.04290233293792
    b = 44.85607552392248
    x_dqn, dqn = svg_to_data_conv(f, a, b)

# with open("Station count.svg") as f:
#     a = -0.1073345259391771
#     b = 52.92486583184257
#     x_stations, stations = svg_to_data_conv(f, a, b, thresh=-0.1)
#     stations = np.interp(x_data, x_stations, stations)

x_station, stations = generate_station_data(x_dqn, 50)

# fig, ax1 = plt.subplots()
# plt.xlabel("Simulation time [s]")

# color = 'tab:blue'
# ax1.set_ylabel('CW', color=color)
# ax1.plot(x_data, data)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()
# color = 'tab:red'
# ax2.set_ylabel('Transmitting stations', color=color)
# ax2.plot(x_station, stations, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# plt.xticks(np.arange(7)*7/6, np.arange(7)*10)
# plt.show()

##### BEB vs DQN throughput
plt.plot(x_beb, beb)
plt.plot(x_dqn, dqn)
plt.ylabel("Throughput [Mb/s]")
plt.xlabel("Simulation time [s]")
plt.xticks(np.arange(7)*7/6, np.arange(7)*10)
plt.show()

# fig, ax1 = plt.subplots()
# plt.xlabel("Simulation time [s]")

# color = 'tab:blue'
# ax1.set_ylabel('Throughput [MB/s]', color=color)
# ax1.plot(x_beb, beb)
# ax1.plot(x_dqn, dqn)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()
# color = 'tab:red'
# ax2.set_ylabel('Transmitting stations', color=color)
# ax2.plot(x_station, stations, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# plt.xticks(np.arange(7)*7/6, np.arange(7)*10)
# plt.show()