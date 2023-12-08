# Neurally Weighted Generalized Cross Correlation (NW-GCC)

import torch.nn as nn

from .signal_processing import GCC
from .layers import Mlp


class NwGCC(GCC):
    """Compute the Neurally Weighted Generalized Cross Correlation (NW-GCC) of the inputs.
    The NW-GCC assigns a likelihood value between 0-1 to the generalized cross correlation,
    learned using a neural network.
    """

    def __init__(self, K, tau_max=None, transform=None, concat_bins=False, center=False):
        super().__init__(K, tau_max, transform, concat_bins, center)

        self.net = Mlp(
            in_features=2 * self.tau_max,
            out_features=1,
            hidden_features=2 * self.tau_max,
            num_layers=2,
            activation=nn.ReLU(),
            batch_norm=False,
            output_activation=nn.Sigmoid(),
        ) # TODO: make this configurable
    
    def forward(self, x, tau_max=None):

        gcc = super().forward(x)

        # gcc /= gcc.abs().max(dim=-1, keepdim=True)[0]

        batch_size, n_frames, n_mics, _, _ = gcc.shape

        weight = self.net(gcc)

        return gcc*weight
