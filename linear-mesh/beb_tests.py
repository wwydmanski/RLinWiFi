
#%%
from ns3gym import ns3env
from comet_ml import Experiment, Optimizer
import tqdm
import subprocess
from collections import deque
import numpy as np
from preprocessor import Preprocessor

from agents.teacher import Teacher, EnvWrapper

class Agent:
    TYPE = "continuous"
    NAME = "STATIC"
    # NAME = "BEB"
    def __init__(self, action_space):
        self.action_space = action_space
        self.actor_loss = 0
        self.critic_loss = 0
        self.lookup = {5: 32, 10:64, 15:128, 25:256}
        self.current_cw = 32

    def act(self, stations_count, *args):
        # return np.random.sample(self.action_space)\
        stations_count = int(stations_count)
        if stations_count in self.lookup.keys():
            self.current_cw = self.lookup[stations_count]

        res = np.array([[np.log2(self.current_cw)-4]])
        return res

    def step(self, *args):
        pass

    def reset(self):
        pass

    def get_loss(self):
        return {"loss": 0}
    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            def foo(*args):
                pass
            return foo

#%%
scenario = "convergence"

simTime = 60 # seconds
stepTime = 0.01  # seconds
history_length = 300

EPISODE_COUNT = 1
steps_per_ep = int(simTime/stepTime)

sim_args = {
    "simTime": simTime,
    "envStepTime": stepTime,
    "historyLength": history_length,
    "agentType": Agent.TYPE,
    "scenario": "basic",
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

tags = [f"{Agent.NAME}", 
        "Final",
        sim_args['scenario'],
        f"Station count: {sim_args['nWifi']}",
        *[f"{key}: {sim_args[key]}" for key in list(sim_args)[:3]]]

#%%
teacher = Teacher(env, 1, Preprocessor(False))
agent = Agent(env.action_space)
logger = teacher.eval(agent, simTime, stepTime, history_length, tags)
