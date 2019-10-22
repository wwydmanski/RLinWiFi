import matplotlib.pyplot as plt
import numpy as np
from comet_ml.papi import API
comet_api = API("hLbeXrPuj2u7JD6QGop7Yrtek")
res = comet_api.get_metrics_for_chart(["ec516b674f594c9485bb2b5b757fa9b8"], ["Chosen CW"])

res = res["ec516b674f594c9485bb2b5b757fa9b8"]['metrics'][0]
steps = np.asarray(res['steps'])
CW = np.asarray(res['values'])

# DQN
dqn_vals = [247, 151, 540, 587, 231, 237, 244, 218, 232, 253, 253, 234, 244, 243, 242]
ddpg_vals = [0, 0, 215, 222, 173, 275, 171, 177, 170, 269, 251, 226, 210, 208]

def get_lim(steps, low, high):
    low_bound = np.where(steps>low)[0][0]
    high_bound = np.where(steps<high)[0][-1]
    return low_bound, high_bound

for i in range(2, 15):
    low, high = get_lim(steps, 6000*i, 6000*(i+1))
    x_beb = steps[low:high]
    beb = CW[low:high]

    res = 0
    for j in range(len(beb)-1):
        res += beb[j+1]*(x_beb[j+1]-x_beb[j])
    res = res/(x_beb[-1]-x_beb[0])
    print(f"{i+1} ({steps[low]}-{steps[high]}): {res}")


plt.plot([i for i in range(len(dqn_vals))], dqn_vals, '-v', markersize=8, label="DQN")
plt.xlabel("Rounds")
plt.ylabel("Mean CW")
plt.legend()
# plt.show()
# def svg_to_data_conv(svg, a, b, thresh=0):
#     bs = BeautifulSoup(f)
#     vals = bs.find_all("path")[-4]["d"].split("L")[1:]
#     data = []
#     unique = []
#     xs = []
#     for val in vals:
#         x = float(val.split(",")[0])/100
#         y_val = float(val.split(",")[1])
#         if x>thresh:
#             if y_val not in unique:
#                 unique.append(y_val)
#             xs.append(x)
#             data.append(a*y_val + b)
#     print(np.max(unique), np.min(unique))
#     return xs, data

# with open("ending_cw_dqn/Chosen CW(9).svg") as f:
#     a = -2.382978
#     b = 1080
#     x_beb, beb = svg_to_data_conv(f, a, b, -0.4)

# stop = 20
# beb = np.asarray(beb)
# plt.plot(x_beb[:stop], beb[:stop])
# # plt.ylim([0, 1024])
# plt.show()
# # print(np.mean(beb[:40]))
# res = 0
# for i in range(stop):
#     print(beb[i+1], x_beb[i+1], x_beb[i])
#     res += beb[i+1]*(x_beb[i+1]-x_beb[i])

# print(res/np.max(x_beb[:stop+1]))