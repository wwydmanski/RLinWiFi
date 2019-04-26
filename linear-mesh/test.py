import numpy as np


def signal_to_stats(signal):
    window = len(signal)//2
    res = []
    for i in range(0, len(signal), window//2):
        res.append([
            [np.mean(signal[i:i+window, batch, 0]),
             np.std(signal[i:i+window, batch, 0]),
             np.mean(signal[i:i+window, batch, 1]),
             np.std(signal[i:i+window, batch, 1])] for batch in range(0, signal.shape[1])])

    return np.array(res)


a = np.random.rand(50, 1, 2)
res = signal_to_stats(a)
print(a)
print(res)
print(res.shape)

print(np.ceil(50/(50//4)).astype(int))