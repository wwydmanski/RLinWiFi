import matplotlib as mpl
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np

plt.rcParams.update({'font.size': 14})

# Avoid Type 3 fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import cycler
n = 2 #number of lines
color = plt.cm.Blues(np.linspace(0.4, 0.7,n)) #gnuplot - Blues name, linspace parameters determine the boundaries of the color
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

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

#plt.figure(figsize=(6.4, 4.8))
fig, ax1 = plt.subplots(figsize=(6.4, 4.8))
plt.xlabel("Simulation time [s]")
ax1.set_ylabel('CW [slots]')
#plt.plot(x_dqn, dqn, label="CCOD w/ DQN", color="#d38fc5")
#ax1.plot(x_data, data, label="CCOD w/ DDPG", color="#2b6f39")
plt.plot(x_dqn, dqn, label="CCOD w/ DQN")
ax1.plot(x_data, data, label="CCOD w/ DDPG")

ax1.tick_params(axis='y')
leg=plt.legend(title='CW', frameon=False)
# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)

ax2 = ax1.twinx()
color = '#000000'
ax2.set_ylabel('Number of stations', color=color)
ax2.plot(x_station, stations, color=color, ls='-', label="Number of stations")
ax2.tick_params(axis='y', labelcolor=color)
leg=ax2.legend(loc='lower right', frameon=False)
# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)

plt.xticks(np.arange(7)*7/6, np.arange(7)*10)
plt.xlim([0, 7])
fig = plt.gcf()
fig.set_size_inches(6.4, 4.8)

plt.savefig('../cw_choice.pdf', bbox_inches='tight')
plt.show()
