import matplotlib as mpl
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np

plt.rcParams.update({'font.size': 14})

# Avoid Type 3 fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import cycler
n = 3 #number of lines
color = plt.cm.Blues(np.linspace(0.3, 1.1,n)) #gnuplot - Blues name, linspace parameters determine the boundaries of the color
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

with open("CT_beb.svg") as f:
    a = -0.04685579196217462
    b = 47.5011111111111
    x_beb, beb = svg_to_data_conv(f, a, b)

with open("CT_ddpg.svg") as f:
    a = -0.02075289575289574
    b = 44.26583011583012
    x_ddpg, ddpg = svg_to_data_conv(f, a, b)

with open("CT_dqn.svg") as f:
    a = -0.03226629256472377
    b = 45.80444841219228
    x_dqn, dqn = svg_to_data_conv(f, a, b)

##### BEB vs DQN throughput
plt.figure(figsize=(6.4, 4.8))
#plt.plot(x_beb, beb, label="Standard 802.11", color="#000000")
#plt.plot(x_ddpg, ddpg, label="CCOD w/ DDPG", color="#f7b051")
#plt.plot(x_dqn, dqn, label="CCOD w/ DQN", color="#e53f26")
plt.plot(x_beb, beb, label="Standard 802.11")
plt.plot(x_dqn, dqn, label="CCOD w/ DQN")
plt.plot(x_ddpg, ddpg, label="CCOD w/ DDPG")

plt.ylabel("Network throughput [Mb/s]")
plt.xlabel("Simulation time [s]")
plt.xticks(np.arange(7)*7/6, np.arange(7)*10)
plt.xlim([0, 7])
plt.ylim([26, 42])
plt.yticks(range(26,44,2))

leg=plt.legend(frameon=False)
# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)
    
plt.savefig('../reaction.pdf', bbox_inches='tight')
plt.show()