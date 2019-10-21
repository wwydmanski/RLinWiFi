
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
    def __init__(self, action_space):
        self.action_space = action_space
        self.actor_loss = 0
        self.critic_loss = 0
        
    def act(self, *args):
        # return np.random.sample(self.action_space)
        res = np.array([[1]])
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

simTime = 30 # seconds
stepTime = 0.01  # seconds
history_length = 3

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
        sim_args['scenario'],
        f"Station count: {sim_args['nWifi']}",
        *[f"{key}: {sim_args[key]}" for key in list(sim_args)[:3]]]

#%%        
teacher = Teacher(env, 1, Preprocessor(False))
agent = Agent(env.action_space)
logger = teacher.eval(agent, simTime, stepTime, history_length, tags)
