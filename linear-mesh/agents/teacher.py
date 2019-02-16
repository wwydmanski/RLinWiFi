import numpy as np
import tqdm
import matplotlib.pyplot as plt
import subprocess
from comet_ml import Experiment

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

    def train(self, EPISODE_COUNT, simTime, stepTime, *tags):
        steps_per_ep = int(simTime/stepTime)
        script_exec_command = f'../../waf --run "linear-mesh --simTime={simTime} --envStepTime={stepTime}"'
        experiment = Experiment(api_key="OZwyhJHyqzPZgHEpDFL1zxhyI",
                        project_name="rl-in-wifi", workspace="wwydmanski")

        if tags is not None:
            experiment.add_tags(tags)

        experiment.log_parameter("Episode count", EPISODE_COUNT)
        experiment.log_parameter("Steps per episode", steps_per_ep)

        for i in range(EPISODE_COUNT):
            if not self.SCRIPT_RUNNING:
                subprocess.Popen(['bash', '-c', script_exec_command])
                self.SCRIPT_RUNNING = True

            cumulative_reward = 0
            reward = 0
            sent_mb = 0

            obs = self.env.reset()

            with tqdm.trange(steps_per_ep) as t:
                for step in t:
                    actions = self.agent.act(np.array([obs], dtype=np.float32), True)
                    next_obs, reward, done, info = self.env.step(actions)
                    self.agent.step(obs, actions, reward,
                                    next_obs, done)
                    obs = next_obs  
                    cumulative_reward += reward

                    sent_mb += float(info)

                    experiment.log_metric("Round reward", reward, step=i*steps_per_ep+step)
                    experiment.log_metric("Per-ep reward", cumulative_reward, step=i*steps_per_ep+step)
                    experiment.log_metric("Megabytes sent", sent_mb, step=i*steps_per_ep+step)
                    experiment.log_metric("Chosen CW", 2**actions[0][0], step=i*steps_per_ep+step)

                    t.set_postfix(mb_sent=f"{sent_mb:.2f} Mb")

            self.agent.reset()
            experiment.log_metric("Cumulative reward", cumulative_reward, step=i)
            experiment.log_metric("actor_loss", self.agent.actor_loss, step=i)
            experiment.log_metric("critic_loss", self.agent.critic_loss, step=i)
            experiment.log_metric("Speed", sent_mb/simTime, step=i)

            print(f"Sent {sent_mb:.2f} Mb/s.\tMean per round: {sent_mb/steps_per_ep:.2f} Mb/s\tEpisode {i+1}/{EPISODE_COUNT} finished\n")
            self.SCRIPT_RUNNING = False

        experiment.end()

        print("Training finished.")