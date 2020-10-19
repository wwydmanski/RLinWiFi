
#%%
from ns3gym import ns3env
from comet_ml import Experiment, Optimizer
import tqdm
import subprocess
from collections import deque
import numpy as np
from preprocessor import Preprocessor
import sys
import argparse

from agents.teacher import Teacher, EnvWrapper

parser = argparse.ArgumentParser(description='Run BEB tests')
parser.add_argument('stations', metavar='N', type=int, nargs='+',
                   help='number of stations for the scenario (min: 5)')
parser.add_argument('--scenario', dest='scenarios', type=str, nargs='+',
                   help='scenarios to run (available: [basic, convergence])')
parser.add_argument('--beb', dest='beb', action='store_const',
                   const=True, default=False,
                   help='run 802.11 default instead of look-up table')

args = parser.parse_args()
for sc in args.scenarios:
    if sc != "basic" and sc!="convergence":
        print("Wrong scenario!")
        parser.print_help()
        sys.exit()

print("Stations count:", args.stations)
print("Scenarios:", args.scenarios)
if args.beb:
    print("802.11 default")
else:
    print("Look-up table")

class Agent:
    TYPE = "continuous"
    NAME = "STATIC"
    # NAME = "BEB"
    def __init__(self, action_space):
        self.action_space = action_space
        self.actor_loss = 0
        self.critic_loss = 0
        self.lookup = {5: 32, 10:64, 15:128, 25:256}
        self.lookup = {}
        for i in range(5, 10):
            self.lookup[i] = 32
        for i in range(10, 15):
            self.lookup[i] = 64
        for i in range(15, 25):
            self.lookup[i] = 128
        for i in range(25, 51):
            self.lookup[i] = 256
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
# scenario = "convergence"

simTime = 60 # seconds
stepTime = 0.01  # seconds
history_length = 300

EPISODE_COUNT = 1
steps_per_ep = int(simTime/stepTime)

if args.beb:
    agent_name = "BEB"
else:
    agent_name = Agent.NAME

rng = 0
for scenario in args.scenarios:
    nwifi = args.stations
    if scenario=="convergence":
        if nwifi[0] == 5:
            nwifi[0] = 6
    
    for nw in nwifi:
        sim_args = {
            "simTime": simTime,
            "envStepTime": stepTime,
            "historyLength": history_length,
            "agentType": Agent.TYPE,
            "scenario": scenario,
            "nWifi": nw,
            "seed": np.random.randint(2**16),
            "rng": rng
        }
        rng += 1
        if args.beb:
            sim_args["dryRun"] = True

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

        tags = [f"{agent_name}", 
                "Final",
                sim_args['scenario'],
                f"Station count: {sim_args['nWifi']}",
                *[f"{key}: {sim_args[key]}" for key in list(sim_args)[:3]]]

        #%%
        teacher =   Teacher(env, 1, Preprocessor(False))
        agent = Agent(env.action_space)
        logger = teacher.eval(agent, simTime, stepTime, history_length, tags)
