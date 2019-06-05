
#%%
from ns3gym import ns3env
from comet_ml import Experiment, Optimizer
import tqdm
import subprocess
from collections import deque
import numpy as np

from agents.ddpg.agent import Agent, Config
from agents.teacher import Teacher, EnvWrapper

#%%
scenario = "convergence"

simTime = 30 # seconds
stepTime = 0.05  # seconds
history_length = 300

EPISODE_COUNT = 30
steps_per_ep = int(simTime/stepTime)

sim_args = {
    "simTime": simTime,
    "envStepTime": stepTime,
    "historyLength": history_length,
    "agentType": Agent.TYPE,
    "scenario": "convergence",
    "nWifi": 15
}
print("Steps per episode:", steps_per_ep)

threads_no = 1
env = EnvWrapper(threads_no, **sim_args)

#%%
env.reset()
ob_space = env.observation_space
ac_space = env.action_space

print("Observation space shape:", ob_space)
print("Action space shape:", ac_space)

assert ob_space is not None

#%%
optimizer = Optimizer("OZwyhJHyqzPZgHEpDFL1zxhyI")

params = """
lr_actor real [1e-6, 5e-3] [3e-5] log
lr_critic real [1e-5, 5e-2] [4e-5] log
"""
optimizer.set_params(params)

#%%
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

class Preprocessor:
    def __init__(self, plot=False):
        self.plot = plot
        if plot:
            self.fig = plt.figure()
            self.ax = plt.gca()

    def preprocess(self, signal):
        window = max(len(signal)//5, 20)
        res = []

        lowess_0 = [lowess(
                        signal[:, batch, 0],
                        np.array([i for i in range(len(signal[:, batch, 0]))]),
                        frac=0.8,
                        return_sorted=False)
                    for batch in range(0, signal.shape[1])]

        lowess_1 = [lowess(
                        signal[:, batch, 1],
                        np.array([i for i in range(len(signal[:, batch, 1]))]),
                        frac=0.8,
                        return_sorted=False)
                    for batch in range(0, signal.shape[1])]

        for i in range(0, len(signal), window//2):
            res.append([
                [np.mean(lowess_0[batch][i:i+window]),
                np.std(lowess_0[batch][i:i+window]),
                np.mean(lowess_1[batch][i:i+window]),
                np.std(lowess_1[batch][i:i+window])] for batch in range(0, signal.shape[1])])
        res = np.array(res)

        if self.plot:
            self.ax.clear()
            self.ax.plot(np.array([i for i in range(len(signal[:, 0, 1]), 0, -1)]), signal[:, 0, 1], c='b')
            self.ax.plot(np.array([i for i in range(len(lowess_1[0]), 0, -1)]), lowess_1[0], c='r')
            self.ax.plot(np.array([i for i in range(len(res[:, 0, 2]), 0, -1)]), res[:, 0, 2], c='g')

            plt.pause(0.001)
        return res

#%%
teacher = Teacher(env, 1, Preprocessor(True))

while True:
    suggestion = optimizer.get_suggestion()

    actor_l = [64, 32, 16]
    critic_l = [64, 32, 16]

    lr_actor = suggestion["lr_actor"]
    lr_critic = suggestion["lr_critic"]

    config = Config(buffer_size=4*steps_per_ep*threads_no, batch_size=128, gamma=0.98, tau=1e-3, lr_actor=lr_actor, lr_critic=lr_critic, update_every=1)

    print("Params:")
    for k, v in suggestion.params.items():
        print(f"{k}: {v}")

    agent = Agent(history_length, action_size=1, config=config, actor_layers = actor_l, critic_layers = critic_l)

    # Test the model
    hyperparams = {**config.__dict__, **sim_args}
    tags = ["Rew: normalized speed", 
            "DDPG", f"Actor: {actor_l}", 
            f"Critic: {critic_l}", 
            f"Instances: {threads_no}",                                 
            f"Station count: {sim_args['nWifi']}", 
            *[f"{key}: {sim_args[key]}" for key in list(sim_args)[:3]]]
    
    logger = teacher.train(agent, EPISODE_COUNT, 
                            simTime=simTime, 
                            stepTime=stepTime, 
                            history_length=history_length, 
                            send_logs=True,
                            experimental=True, 
                            tags=tags,
                            parameters=hyperparams)

    # Report the score back
    suggestion.report_score("last_speed", logger.last_speed)
    del agent