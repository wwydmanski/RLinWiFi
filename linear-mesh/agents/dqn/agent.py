import numpy as np
import random
from collections import namedtuple, deque
import datetime
import tensorflow as tf

from .model import QNetworkTf


# BUFFER_SIZE = int(1e5)  # replay buffer size
# BATCH_SIZE = 32        # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 1e-3              # for soft update of target parameters
# LR = 5e-4               # learning rate
# UPDATE_EVERY = 2        # how often to update the network

E = 0.01
A = 0.6
B = 0.4


class Config:
    def __init__(self, buffer_size=int(2e3), batch_size=64, gamma=0.99, tau=1e-3, lr=5e-4, update_every=4):
        self.BUFFER_SIZE = int(buffer_size)
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau
        self.LR = lr
        self.UPDATE_EVERY = update_every


class Agent:
    """Interacts with and learns from the environment."""
    TYPE = "discrete"
    NAME = "DQN"

    def __del__(self):
        tf.reset_default_graph()

    def __init__(self, network, state_size, action_size, config=Config(), seed=42, save=True, save_loc='models/', save_every=2, checkpoint_file=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.args = (network, state_size, action_size,
                     seed, save, save_loc, save_every)
        self.config = config

        self.sess = tf.Session()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = network(
            self.sess, state_size, action_size, "local", self.config.LR, checkpoint_file)
        self.qnetwork_target = network(
            self.sess, state_size, action_size, "target", self.config.LR, checkpoint_file)

        self.memory = ReplayBuffer(
            action_size, self.config.BUFFER_SIZE, self.config.BATCH_SIZE, seed)
        self.t_step = 0

        self.soft_update_op = self._get_soft_update_op()

        self.loss = 0
        self.learning_steps = 0

        self.eps = tf.Variable(0.0, trainable=False)
        self.eps_end = 0.0
        self.eps_decay = 0.0

        if checkpoint_file is None:
            self.saver = tf.train.Saver()
            self.save_config = {'save': save,
                                'save_loc': save_loc, 'save_every': save_every}
        else:
            self.saver = None
            self.save_config = None

        self.episode_counter = 0
        self.notifications = 0

        print("Action space:", action_size)

        self.act_op = tf.cond(tf.random_uniform([1], dtype=tf.float32)[0] > self.eps,
                              lambda: tf.argmax(
                                  self.qnetwork_local.output, output_type=tf.int32, axis=1),
                              lambda: tf.random_uniform([1], minval=0, maxval=action_size, dtype=tf.int32))
        self.no_noise_act_op = tf.argmax(self.qnetwork_local.output, output_type=tf.int32, axis=1)

        self.sess.run([tf.local_variables_initializer(),
                       tf.global_variables_initializer()])

    def load(self):
        pass

    def reset_all(self):
        self.sess.close()
        tf.reset_default_graph()
        self.__init__(*self.args)

    def set_epsilon(self, epsilon_start, epsilon_end, episodes_till_cap):
        self.eps.load(epsilon_start, self.sess)
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_end**(1/episodes_till_cap)

    def reset(self):
        self.episode_counter += 1
        self.eps.load(max(self.eps_end, self.eps_decay *
                          self.eps.eval(self.sess)), self.sess)

        if self.saver is not None:
            if self.save_config['save'] and self.episode_counter % self.save_config['save_every'] == 0:
                now = datetime.datetime.now()
                self.saver.save(
                    self.sess, self.save_config["save_loc"]+now.strftime("%Y-%m-%d_%H-%M.ckpt"))

    def _get_soft_update_op(self):
        Qvars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='inference_local')
        target_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='inference_target')

        return [tvar.assign(self.config.TAU*qvar + (1.0-self.config.TAU)*tvar) for qvar, tvar in zip(Qvars, target_vars)]

    def step(self, state, action, reward, next_state, done, iter_count=1):
        """ Stores SARS in memory for further processing and teaches agent based

        Args:
            state (array_like): state before taking action
            action (int): taken action
            reward (float): reward for taking action
            next_state (array_like): state after taking action
            done (bool): whether action ended episode
        """
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.config.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.config.BATCH_SIZE:
                if self.notifications == 0:
                    print("------- STARTED TRAINING -------")
                    self.notifications = 1
                elif self.notifications == 1 and len(self.memory) == self.config.BUFFER_SIZE:
                    print("------- MEMORY BUFFER FILLED -------")
                    self.notifications = -1
                for i in range(iter_count):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.config.GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            add_noise (bool): false for greedy strategy
        """
        res = []
        for sim_id in range(state.shape[1]):
            sim = np.expand_dims(state[:, sim_id], 1)
            if add_noise:
                res.append(self.sess.run(self.act_op, feed_dict={self.qnetwork_local.input: sim}))
            else:
                res.append(self.sess.run(self.no_noise_act_op, feed_dict={self.qnetwork_local.input: sim}))

        return res

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        q_target_output = self.qnetwork_target.forward(next_states)
        Q_targets_next = np.expand_dims(
            np.amax(q_target_output, 1), 1)
        Q_targets = rewards + (gamma*Q_targets_next*(1-dones))

        reduced_loss, result = self.qnetwork_local.train(
            states, Q_targets, actions)
        self._update_loss(reduced_loss)

        self.soft_update()

    def soft_update(self):
        self.sess.run(self.soft_update_op)

    def _update_loss(self, loss):
        self.loss = self.loss*self.learning_steps / \
            (self.learning_steps+1) + loss/(self.learning_steps+1)
        self.learning_steps += 1

    def get_loss(self):
        return {"loss": self.loss}


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
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        state = np.array(state)
        next_state = np.array(next_state)

        for sim_id in range(state.shape[1]):
            e = self.experience(state[:, sim_id], action[sim_id], reward[sim_id], next_state[:, sim_id], done[sim_id])
            self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.stack([e.state for e in experiences if e is not None], 1)
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.stack(
            [e.next_state for e in experiences if e is not None], 1)
        dones = np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
