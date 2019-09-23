
#%%
from ns3gym import ns3env
from comet_ml import Experiment, Optimizer
import tqdm
import subprocess
from collections import deque
import numpy as np

from agents.teacher import Teacher, EnvWrapper

#%%
scenario = "convergence"

simTime = 60 # seconds
stepTime = 0.01  # seconds
history_length = 300

EPISODE_COUNT = 10
steps_per_ep = int(simTime/stepTime)

sim_args = {
    "simTime": simTime,
    "envStepTime": stepTime,
    "historyLength": history_length,
    "agentType": Agent.TYPE,
    "scenario": "convergence",
    "nWifi": 30
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
        
    def __getattribute__(self, attr):
        def foo(*args):
            pass
        return foo
        
teacher = Teacher(env, 1, Preprocessor(True))
agent = Agent(env.action_space)
logger = teacher.dry_run(agent, int(simTime/stepTime))
