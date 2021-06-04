import torch
import pfrl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pfrl.initializers import init_chainer_default

def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)

    return init_bias

class LargeAtariCNN(nn.Module):
    """Large CNN module proposed for DQN in Nature, 2015.

    See: https://www.nature.com/articles/nature14236
    """

    def __init__(
        self, obs_size, n_input_channels=3, n_output_channels=512, activation=F.relu, bias=0.1
    ):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )
        dummy_data = torch.zeros((1,) + obs_size).transpose(1, 3).transpose(2, 3)
        for l in self.layers:
            dummy_data = l(dummy_data)
        dummy_output = torch.flatten(dummy_data)
        self.output = nn.Linear(dummy_output.size(0), n_output_channels)

        self.apply(init_chainer_default)
        self.apply(constant_bias_initializer(bias=bias))

    def forward(self, state):
        h = state.transpose(1, 3).transpose(2, 3)
        for layer in self.layers:
            h = self.activation(layer(h))
        h_flat = h.reshape(h.size(0), -1)
        return self.activation(self.output(h_flat))

def make_large_atari_q_func(env_name, obs_space, action_space):
    obs_size = obs_space.shape
    return nn.Sequential(
            LargeAtariCNN(obs_size),
            init_chainer_default(nn.Linear(512, action_space.n)),
            pfrl.q_functions.DiscreteActionValueHead()
        )    

def make_large_atari_error_func(env_name, obs_space, action_space):
    obs_size = obs_space.shape
    return nn.Sequential(
            LargeAtariCNN(obs_size),
            init_chainer_default(nn.Linear(512, 512)),
            init_chainer_default(nn.Linear(512, action_space.n)),
            pfrl.q_functions.DiscreteActionValueHead()
        )  

def phi(x):
    return x / 255.0