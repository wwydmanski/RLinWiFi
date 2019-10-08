
#%%
from ns3gym import ns3env
from comet_ml import Experiment, Optimizer
import tqdm
import subprocess
from collections import deque
import numpy as np

from agents.ddpg.agent import Agent, Config
from agents.teacher import Teacher, EnvWrapper
from preprocessor import Preprocessor

#%%
scenario = "convergence"

simTime = 30 # seconds
stepTime = 0.1  # seconds
history_length = 300

EPISODE_COUNT = 1
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
class Agent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.actor_loss = 0
        self.critic_loss = 0
        
    def act(self, *args):
        return np.random.sample(self.action_space)
    
    def step(self, *args):
        pass
    
    def reset(self):
        pass

#%%
teacher = Teacher(env, 1, Preprocessor(True))
agent = Agent(env.action_space)
logger = teacher.dry_run(agent, int(simTime/stepTime))