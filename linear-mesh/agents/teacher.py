import numpy as np
import tqdm
import matplotlib.pyplot as plt
import subprocess
from comet_ml import Experiment

class Logger:
    def __init__(self, *tags, **parameters):
        self.experiment = Experiment(api_key="OZwyhJHyqzPZgHEpDFL1zxhyI",
                        project_name="rl-in-wifi", workspace="wwydmanski")
        self.sent_mb = 0

        if tags is not None:
            self.experiment.add_tags(tags)
        if parameters is not None:
            self.experiment.log_parameters(parameters)

    def begin_logging(self, episode_count, steps_per_ep):
        self.experiment.log_parameter("Episode count", episode_count)
        self.experiment.log_parameter("Steps per episode", steps_per_ep)

    def log_round(self, reward, cumulative_reward, info, loss, step):
        round_mb = float(info.split("|")[0])
        self.sent_mb += round_mb
        CW = float(info.split("|")[1])

        self.experiment.log_metric("Round reward", reward, step=step)
        self.experiment.log_metric("Per-ep reward", cumulative_reward, step=step)
        self.experiment.log_metric("Megabytes sent", self.sent_mb, step=step)
        self.experiment.log_metric("Round megabytes sent", round_mb, step=step)
        self.experiment.log_metric("Chosen CW", CW, step=step)
        self.experiment.log_metrics(loss, step=step)

    def log_episode(self, cumulative_reward, speed, step):
        self.experiment.log_metric("Cumulative reward", cumulative_reward, step=step)
        self.experiment.log_metric("Speed", speed, step=step)

        self.sent_mb = 0

    def end(self):
        self.experiment.end()

class Teacher:
    """Class that handles training of RL model in ns-3 simulator

    Attributes:
        agent: agent which will be trained
        env (ns3-gym env): environment used for learning. NS3 program must be run before creating teacher
        num_agents (int): number of agents present at once
    """

    def __init__(self, agent, env, num_agents):
        self.agent = agent
        self.env = env
        self.num_agents = num_agents
        self.SCRIPT_RUNNING = True
        self.CW = 16
        self.action = None              # For debug purposes

    def train(self, EPISODE_COUNT, simTime, stepTime, script_exec_command, *tags, **parameters):
        steps_per_ep = int(simTime/stepTime)

        logger = Logger(*tags, **parameters)
        logger.begin_logging(EPISODE_COUNT, steps_per_ep)

        for i in range(EPISODE_COUNT):
            if not self.SCRIPT_RUNNING:
                subprocess.Popen(['bash', '-c', script_exec_command])
                self.SCRIPT_RUNNING = True

            cumulative_reward = 0
            reward = 0
            sent_mb = 0

            obs = self.env.reset()
            self.last_actions = None

            with tqdm.trange(steps_per_ep) as t:
                for step in t:
                    self.actions = self.agent.act(np.array([obs], dtype=np.float32), True)
                    next_obs, reward, done, info = self.env.step(self.actions)

                    if self.last_actions is not None:
                        self.agent.step(obs, self.last_actions, reward,
                                        next_obs, done)
                    obs = next_obs  
                    cumulative_reward += reward

                    self.last_actions = self.actions

                    logger.log_round(reward, cumulative_reward, info, self.agent.get_loss(), i*steps_per_ep+step)
                    t.set_postfix(mb_sent=f"{logger.sent_mb:.2f} Mb")

            self.agent.reset()
            print(f"Sent {logger.sent_mb:.2f} Mb/s.\tMean speed: {logger.sent_mb/simTime:.2f} Mb/s\tEpisode {i+1}/{EPISODE_COUNT} finished\n")

            logger.log_episode(cumulative_reward, logger.sent_mb/simTime, i)
            self.SCRIPT_RUNNING = False

        logger.end()

        print("Training finished.")