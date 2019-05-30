import numpy as np
import tqdm
import subprocess
from comet_ml import Experiment
from ns3gym import ns3env
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def signal_to_stats(signal, ax=None):
    window = max(len(signal)//10, 20)
    res = []

    lowess_0 = [lowess(
                    signal[:, batch, 0], 
                    np.array([i for i in range(len(signal[:, batch, 0]))]), 
                    frac=0.4, 
                    return_sorted=False) 
                for batch in range(0, signal.shape[1])]

    lowess_1 = [lowess(
                    signal[:, batch, 1],
                    np.array([i for i in range(len(signal[:, batch, 1]))]), 
                    frac=0.4, 
                    return_sorted=False) 
                for batch in range(0, signal.shape[1])]

    for i in range(0, len(signal), window//2):
        res.append([
            [np.mean(lowess_0[batch][i:i+window]),
             np.std(lowess_0[batch][i:i+window]),
             np.mean(lowess_1[batch][i:i+window]),
             np.std(lowess_1[batch][i:i+window])] for batch in range(0, signal.shape[1])])
    res = np.array(res)

    if ax is not None:
        ax.clear()
        ax.plot(np.array([i for i in range(len(signal[:, 0, 1]), 0, -1)]), signal[:, 0, 1], c='b')
        ax.plot(np.array([i for i in range(len(lowess_1[0]), 0, -1)]), lowess_1[0], c='r')
        ax.plot(np.array([i for i in range(len(res[:, 0, 2]), 0, -1)]), res[:, 0, 2], c='g')

        plt.pause(0.001)
    return res

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

    def log_round(self, reward, cumulative_reward, info, loss, observations, step):

        try:
            round_mb = np.mean([float(i.split("|")[0]) for i in info])
        except Exception as e:
            print(info)
            print(reward)
            raise e
        self.sent_mb += round_mb
        CW = np.mean([float(i.split("|")[1]) for i in info])

        self.experiment.log_metric("Round reward", np.mean(reward), step=step)
        self.experiment.log_metric("Per-ep reward", np.mean(cumulative_reward), step=step)
        self.experiment.log_metric("Megabytes sent", self.sent_mb, step=step)
        self.experiment.log_metric("Round megabytes sent", round_mb, step=step)
        self.experiment.log_metric("Chosen CW", CW, step=step)

        for i, obs in enumerate(observations):
            self.experiment.log_metric(f"Observation {i}", obs, step=step)

        self.experiment.log_metrics(loss, step=step)

    def log_episode(self, cumulative_reward, speed, step):
        self.experiment.log_metric("Cumulative reward", cumulative_reward, step=step)
        self.experiment.log_metric("Speed", speed, step=step)

        self.sent_mb = 0
        self.last_speed = speed

    def end(self):
        self.experiment.end()

class Teacher:
    """Class that handles training of RL model in ns-3 simulator

    Attributes:
        agent: agent which will be trained
        env (ns3-gym env): environment used for learning. NS3 program must be run before creating teacher
        num_agents (int): number of agents present at once
    """

    def __init__(self, env, num_agents):
        self.env = env
        self.num_agents = num_agents
        self.CW = 16
        self.action = None              # For debug purposes

    def train(self, agent, EPISODE_COUNT, simTime, stepTime, history_length, *tags, **parameters):
        # fig = plt.figure()
        # ax = plt.gca()
        steps_per_ep = int(simTime/stepTime)

        logger = Logger(*tags, **parameters)
        logger.begin_logging(EPISODE_COUNT, steps_per_ep)
        add_noise = True

        obs_dim = 2
        time_offset = history_length//2*stepTime

        for i in range(EPISODE_COUNT):
            try:
                self.env.run()
            except AlreadyRunningException as e:
                pass

            if i>=EPISODE_COUNT*2/3:
                add_noise = False

            cumulative_reward = 0
            reward = 0
            sent_mb = 0

            obs = self.env.reset()
            obs = signal_to_stats(np.reshape(obs, (-1, len(self.env.envs), obs_dim)))

            self.last_actions = None

            with tqdm.trange(steps_per_ep) as t:
                for step in t:
                    self.debug = obs

                    self.actions = agent.act(np.array(obs, dtype=np.float32), add_noise)
                    next_obs, reward, done, info = self.env.step(self.actions)

                    next_obs = signal_to_stats(np.reshape(next_obs, (-1, len(self.env.envs), obs_dim)))
                    if self.last_actions is not None and step>(history_length/obs_dim):
                        agent.step(obs, self.last_actions, reward, next_obs, done)
                    obs = next_obs  
                    cumulative_reward += np.mean(reward)

                    self.last_actions = self.actions

                    if step>(history_length/obs_dim):
                        logger.log_round(reward, cumulative_reward, info, agent.get_loss(), obs[0][0][::2], i*steps_per_ep+step)
                    t.set_postfix(mb_sent=f"{logger.sent_mb:.2f} Mb")

            agent.reset()
            self.env.close()
            print(f"Sent {logger.sent_mb:.2f} Mb/s.\tMean speed: {logger.sent_mb/(simTime-time_offset):.2f} Mb/s\tEpisode {i+1}/{EPISODE_COUNT} finished\n")

            logger.log_episode(cumulative_reward, logger.sent_mb/(simTime-time_offset), i)

        logger.end()
        print("Training finished.")
        return logger

class AlreadyRunningException(Exception):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

class EnvWrapper:
    def __init__(self, no_threads, **params):
        self.no_threads = no_threads
        self.ports = [33968+i+np.random.randint(500) for i in range(no_threads)]
        self.commands = self._craft_commands(params)

        self.SCRIPT_RUNNING = False
        self.envs = []

        self.run()
        for port in self.ports:
            env = ns3env.Ns3Env(port=port, stepTime=params['envStepTime'], startSim=0, simSeed=0, simArgs=params, debug=False)
            self.envs.append(env)
        
        self.SCRIPT_RUNNING = True

    def run(self):
        if self.SCRIPT_RUNNING:
            raise AlreadyRunningException("Script is already running")

        for cmd, port in zip(self.commands, self.ports):
            subprocess.Popen(['bash', '-c', cmd])
        self.SCRIPT_RUNNING = True

    def _craft_commands(self, params):
        command = '../../waf --run "linear-mesh'
        for key, val in params.items():
            command+=f" --{key}={val}"
        
        commands = []
        for p in self.ports:
            commands.append(command+f' --openGymPort={p}"')

        return commands

    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())

        return obs
    
    def step(self, actions):
        next_obs, reward, done, info = [], [], [], []

        for i, env in enumerate(self.envs):
            no, rew, dn, inf = env.step(actions[i].tolist())
            next_obs.append(no)
            reward.append(rew)
            done.append(dn)
            info.append(inf)

        return next_obs, reward, done, info

    @property
    def observation_space(self):
        return (self.no_threads, repr(self.envs[0].observation_space))

    @property
    def action_space(self):
        return (self.no_threads, repr(self.envs[0].action_space))

    def close(self):
        for env in self.envs:
            env.close()
        subprocess.Popen(['bash', '-c', "killall linear-mesh"])
        
        self.SCRIPT_RUNNING = False

    def __getattr__(self, attr):
        for env in self.envs:
            env.attr()
