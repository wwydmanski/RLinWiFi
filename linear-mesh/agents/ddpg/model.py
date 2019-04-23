import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=128, fc2_units=64, fc3_units=32):
        """Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int, optional): Defaults to 300. Number of nodes in first hidden layer
            fc2_units (int, optional): Defaults to 200. Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.fc_units = fc_units
        self.state_size = int(state_size/2)
        self.seed = torch.manual_seed(seed)
        self.norm = torch.nn.BatchNorm1d(self.state_size)
        self.lstm1 = nn.LSTM(2, fc_units)
        self.norm1 = torch.nn.BatchNorm1d(fc_units)
        self.fc2 = nn.Linear(fc_units, fc2_units)
        self.norm2 = torch.nn.BatchNorm1d(fc2_units)
        # self.fc3 = nn.Linear(fc2_units, action_size)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

        self.reset_parameters()

    def reset_parameters(self):
        # self.lstm1.weight.data.uniform_(*hidden_init(self.lstm1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))

        # self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        h0 = torch.randn(1, state.size()[0], self.fc_units).cuda()
        c0 = torch.randn(1, state.size()[0], self.fc_units).cuda()

        # for i in state:
        #     x, hidden = self.lstm1(i.view(1, 1, -1), hidden)
        # print(hidden.size())
        inp = state.view(self.state_size, -1, 2)
        x, _ = self.lstm1(inp, (h0, c0))
        x = self.norm1(F.relu(x[-1]))
        x = self.fc2(x)
        x = self.norm2(F.relu(x))
        # return torch.tanh(self.fc3(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=512, fc2_units=256, fc3_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.norm = torch.nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.norm1 = torch.nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, 1)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.fcs1(state)
        xs = self.norm1(F.relu(xs))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # return self.fc3(x)
        return self.fc4(x)
