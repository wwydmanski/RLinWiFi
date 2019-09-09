#%%
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np

class Preprocessor:
    def __init__(self, plot=False):
        self.plot = plot
        if plot:
            self.fig, self.ax = plt.subplots(2, sharex=True)

    def normalize(self, sig):
        # if np.max(sig)-np.min(sig)==0:
            # return np.zeros(sig.shape)

        return np.clip((sig-np.min(sig))/(np.max(sig)-np.min(sig)+1e-6), 0, 1)

    def preprocess(self, signal):
        # window = max(len(signal)//4, 10)
        window = 150
        res = []

        """ lowess_0 = [lowess(
                        signal[:, batch, 0],
                        np.array([i for i in range(len(signal[:, batch, 0]))]),
                        frac=0.2,
                        return_sorted=False)
                    for batch in range(0, signal.shape[1])]"""
        # lowess_1 = [lowess(
        #                 signal[:, batch, 1],
        #                 np.array([i for i in range(len(signal[:, batch, 1]))]),
        #                 frac=0.2,
        #                 return_sorted=False)
        #             for batch in range(0, signal.shape[1])]
        for i in range(0, len(signal), window//2):
            res.append([
                [np.mean(signal[i:i+window, batch]),
                np.std(signal[i:i+window, batch])] for batch in range(0, signal.shape[1])])
        res = np.array(res)
        res = np.clip(res, 0, 1)

        if self.plot:
            plot_len = len(signal[:, 0, 0])
            plot_range = [i for i in range(plot_len, 0, -1)]

            res_0 = self.normalize(res[:, :, 0].repeat(window//2, 0))

            self.ax[0].clear()
            self.ax[0].plot(np.array(plot_range), self.normalize(signal[:, 0, 0]), c='b')
            self.ax[0].plot(np.array(plot_range), res_0, c='g')
            self.ax[0].plot(np.array(plot_range), res_0 + res[:, :, 1].repeat(window//2, 0), c='r')

            self.ax[1].clear()
            self.ax[1].plot(np.array(plot_range), signal[:, 0, 1], c='b')
            self.ax[1].plot(np.array(plot_range), res[:, :, 2].repeat(window//2, 0), c='g')

            plt.pause(0.001)
        return res
