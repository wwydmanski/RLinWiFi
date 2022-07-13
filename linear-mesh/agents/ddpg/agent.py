from .model import Actor, Critic
import numpy as np
import torch
from collections import namedtuple, deque
import random
import copy

# BUFFER_SIZE = int(2e3)  # replay buffer size
# BATCH_SIZE = 64         # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 1e-3              # for soft update of target parameters
# LR_ACTOR = 1e-5         # learning rate of the actor
# LR_CRITIC = 4e-3        # learning rate of the critic
# UPDATE_EVERY = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Config:
    def __init__(self, buffer_size=int(2e3), batch_size=64, gamma=0.99, tau=1e-3, lr_actor=1e-5, lr_critic=4e-3, update_every=4):
        self.BUFFER_SIZE = int(float(buffer_size))      # Fix for scientific notation
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.UPDATE_EVERY = update_every


class Agent:
    TYPE = "continuous"
    NAME = "DDPG"

    def __init__(self, state_size, action_size, config=Config(), random_seed=42, actor_layers=None, critic_layers=None):
        print("CuDNN version:", torch.backends.cudnn.version())
        print("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.action_size = action_size
        self.noise = NormalNoise(action_size, random_seed, mu=0, sigma=4, theta=0.7)

        if actor_layers is None:
            self.actor_local = Actor(
                state_size, action_size, random_seed, config.BATCH_SIZE).to(device)
            self.actor_target = Actor(
                state_size, action_size, random_seed, config.BATCH_SIZE).to(device)
        else:
            self.actor_local = Actor(
                state_size, action_size, random_seed, config.BATCH_SIZE, *actor_layers).to(device)
            self.actor_target = Actor(
                state_size, action_size, random_seed, config.BATCH_SIZE, *actor_layers).to(device)

        if critic_layers is None:
            self.critic_local = Critic(
                state_size, action_size, random_seed, config.BATCH_SIZE).to(device)
            self.critic_target = Critic(
                state_size, action_size, random_seed, config.BATCH_SIZE).to(device)
        else:
            self.critic_local = Critic(
                state_size, action_size, random_seed, config.BATCH_SIZE, *critic_layers).to(device)
            self.critic_target = Critic(
                state_size, action_size, random_seed, config.BATCH_SIZE, *critic_layers).to(device)

        self.critic_loss = 0
        self.actor_loss = 0

        # HARD update
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)

        self.memory = ReplayBuffer(
            action_size, self.config.BUFFER_SIZE, self.config.BATCH_SIZE, random_seed)

        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(), lr=self.config.LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(), lr=self.config.LR_CRITIC)
        self.t_step = 0

        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=5, gamma=0.1)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=5, gamma=0.1)
        self.episodes_passed = 1

        self.notifications = 0

    def set_config(self, config):
        self.config = config

        self.memory = ReplayBuffer(
            action_size, self.config.BUFFER_SIZE, self.config.BATCH_SIZE, 42)

        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(), lr=self.config.LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(), lr=self.config.LR_CRITIC)

        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=5, gamma=0.1)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=5, gamma=0.1)
        self.t_step = 0


    def act(self, state, add_noise):
        """Get action according to actor policy

        Arguments:
            state (List[float]): Current observation of environment
            add_noise (bool): Whether to add noise from Ornstein-Uhlenbeck process

        Returns:
            ndarray[np.float32] -- Estimated best action
        """

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action_values = np.clip(self.actor_local(state).cpu().data.numpy(), 0, 6)
        self.actor_local.train()

        if add_noise:
            for i in range(action_values.shape[0]):
                action_values[i] += self.noise.sample()
                # action_values[i] += (self.noise.sample()-0.8) / \
                #     np.sqrt(self.episodes_passed)

        return np.clip(action_values, 0, 6)

    def step(self, states, actions, rewards, next_states, dones, training_steps=1):
        for action, reward, done, i in zip(actions, rewards, dones, range(len(rewards))):
            assert states[:, i].ndim==2
            assert next_states[:, i].ndim==2
            self.memory.add(states[:, i], action, reward, next_states[:, i], done)

        self.t_step += 1

        if self.t_step % self.config.UPDATE_EVERY == 0:
            if len(self.memory) > self.config.BATCH_SIZE:
                if self.notifications == 0:
                    print("------- STARTED TRAINING -------")
                    self.notifications = 1
                elif self.notifications == 1 and len(self.memory) == self.config.BUFFER_SIZE:
                    print("------- MEMORY BUFFER FILLED -------")
                    self.notifications = -1

                experiences = self.memory.sample()
                for i in range(training_steps):
                   self.learn(experiences, self.config.GAMMA)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples

        Arguments:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        max_Qhat = self.critic_target(
            next_states, self.actor_target(next_states))

        Q_target = rewards + (gamma * max_Qhat * (1 - dones))

        Q_expected = self.critic_local(states, actions)
        loss = torch.nn.functional.mse_loss(Q_expected, Q_target)
        self.critic_loss = loss.cpu().data.numpy()

        self.critic_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        loss.backward()
        self.critic_optimizer.step()

        policy_loss = -self.critic_local(states, self.actor_local(states))
        policy_loss = policy_loss.mean()
        self.actor_loss = policy_loss.cpu().data.numpy()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.actor_local, self.actor_target, self.config.TAU)
        self.soft_update(self.critic_local,
                         self.critic_target, self.config.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Arguments:
            local_model (PyTorch model): model weights will be copied from
            target_model (PyTorch model): model weights will be copied to
            tau (float): interpolation parameter
        """
        iter_params = zip(target_model.parameters(), local_model.parameters())
        for target_param, local_param in iter_params:
            tensor_aux = tau*local_param.data + (1.0-tau)*target_param.data
            target_param.data.copy_(tensor_aux)

    def reset(self):
        self.noise.reset()
        self.episodes_passed += 1
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def get_loss(self):
        return {"actor_loss": self.actor_loss, "critic_loss": self.critic_loss}

    def reset_parameters(self):
        self.actor_local.reset_parameters()
        self.actor_target.reset_parameters()
        self.critic_local.reset_parameters()
        self.critic_target.reset_parameters()
        self.memory = ReplayBuffer(
            self.action_size, self.config.BUFFER_SIZE, self.config.BATCH_SIZE, 42)

        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(), lr=self.config.LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(), lr=self.config.LR_CRITIC)
        self.t_step = 0

        self.episodes_passed = 1

        self.notifications = 0

    def save(self):
        torch.save(self.actor_local.state_dict(), "models/ddpg_actor.torch")
        torch.save(self.critic_local.state_dict(), "models/ddpg_critic.torch")

    def load(self):
        try:
            self.actor_local.load_state_dict(torch.load("models/ddpg_actor_15_convergence.torch"))
            self.critic_local.load_state_dict(torch.load("models/ddpg_critic_15_convergence.torch"))
        except RuntimeError:
            self.actor_local.load_state_dict(torch.load("models/ddpg_actor_15_convergence.torch", map_location='cpu'))
            self.critic_local.load_state_dict(torch.load("models/ddpg_critic_15_convergence.torch", map_location='cpu'))            

class NormalNoise:
    def __init__(self, size, seed, mu=0., sigma=0.2, theta=0.6):
        """Initialize parameters and noise proces:

        Arguments:
            size (int): number of output values
            seed (int): disregarded
            mu (float): mean of values
            sigma (float): standard deviation
            theta (float): rate of sigma diminishing
          """
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.theta = theta

    def reset(self):
        """Reduce sigma"""
        self.sigma *= self.theta

    def sample(self):
        return np.random.normal(self.mu, self.sigma, self.size)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        if seed!=-1:
            self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.stack([e.state for e in experiences if e is not None], 1)).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack(
            [e.next_state for e in experiences if e is not None], 1)).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
