
#%%
from ns3gym import ns3env
from comet_ml import Experiment, Optimizer
import tqdm
import subprocess
from collections import deque
import numpy as np

from agents.dqn.agent import Agent, Config
from agents.dqn.model import QNetworkTf
from agents.teacher import Teacher, EnvWrapper
from preprocessor import Preprocessor

#%%
scenario = "convergence"

simTime = 60 # seconds
stepTime = 0.03  # seconds
history_length = 210

EPISODE_COUNT = 15
steps_per_ep = int(simTime/stepTime)

sim_args = {
    "simTime": simTime,
    "envStepTime": stepTime,
    "historyLength": history_length,
    "agentType": Agent.TYPE,
    "scenario": "convergence",
    "nWifi": 6
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
# optimizer = Optimizer("OZwyhJHyqzPZgHEpDFL1zxhyI")

# params = """
# lr_actor real [1e-6, 5e-3] [1e-5] log
# lr_critic real [1e-5, 5e-2] [4e-5] log
# """
# optimizer.set_params(params)
config = {
  "algorithm": "bayes",
  "parameters": {
      "lr": {"type": "float", "min": 1e-4, "max": 1e-2 },
  },
  "spec": {
      "metric": "Speed",
      "objective": "maximize",
  },
}
optimizer = Optimizer(config, api_key="OZwyhJHyqzPZgHEpDFL1zxhyI",
                    project_name="rl-in-wifi")

#%%
teacher = Teacher(env, 1, Preprocessor(False))

# actor_l = [64, 32, 16]
# critic_l = [64, 32, 16]

# lr_actor = 1e-5
# lr_critic = 4e-5

for experiment in optimizer.get_experiments():
    lr = experiment.get_parameter("lr")
    # config = Config(buffer_size=4*steps_per_ep*threads_no, batch_size=256, gamma=0.98, tau=1e-3, lr_actor=lr_actor, lr_critic=lr_critic, update_every=1)

    config = Config(buffer_size=4*steps_per_ep*threads_no, batch_size=32, gamma=0.98, tau=1e-3, lr=lr, update_every=1)
    agent = Agent(QNetworkTf, history_length, action_size=3, config=config)
    agent.set_epsilon(0.9, 0.01, 3)
    experiment.log_metric("learning_rate", lr, step=int(simTime/stepTime)*5)
    # agent = Agent(history_length, action_size=3, config=config)

    # Test the model
    hyperparams = {**config.__dict__, **sim_args}
    tags = ["Rew: normalized speed",
            f"{Agent.NAME}",
            f"LR: {lr}",
            # f"Actor: {actor_l}",
            # f"Critic: {critic_l}",
            f"Instances: {threads_no}",
            f"Station count: {sim_args['nWifi']}",
            *[f"{key}: {sim_args[key]}" for key in list(sim_args)[:3]]]

    logger = teacher.train(agent, 5,
                            simTime=simTime,
                            stepTime=stepTime,
                            history_length=history_length,
                            send_logs=True,
                            experimental=True,
                            tags=tags,
                            parameters=hyperparams,
                            experiment=experiment)
    experiment.end()
    del agent
