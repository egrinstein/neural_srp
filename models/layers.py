"""
	Pytorch functions and layers for DOA estimation.

	File name: acousticTrackingModules.py
	Author: David Diaz-Guerra
	Date creation: 05/2020
	Python Version: 3.8
	Pytorch Version: 1.4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausConv3d(nn.Module):
    """Causal 3D Convolution for SRP-PHAT maps sequences"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.pad = kernel_size[0] - 1
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, padding=(self.pad, 0, 0)
        )

    def forward(self, x):
        return self.conv(x)[:, :, : -self.pad, :, :]


class CausConv2d(nn.Module):
    """Causal 2D Convolution for spectrograms and GCCs sequences"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.pad = kernel_size[0] - 1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=(self.pad, 0)
        )

    def forward(self, x):
        return self.conv(x)[:, :, : -self.pad, :]


class CausConv1d(nn.Module):
    """Causal 1D Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation
        )

    def forward(self, x):
        return self.conv(x)[:, :, : -self.pad]


class SphericPad(nn.Module):
    """Replication padding for time axis, reflect padding for the elevation and circular padding for the azimuth.
    The time padding is optional, do not use it with CausConv3d.
    """

    def __init__(self, pad):
        super().__init__()

        if len(pad) == 4:
            self.padLeft, self.padRight, self.padTop, self.padBottom = pad
            self.padFront, self.padBack = 0, 0
        elif len(pad) == 6:
            (
                self.padLeft,
                self.padRight,
                self.padTop,
                self.padBottom,
                self.padFront,
                self.padBack,
            ) = pad
        else:
            raise Exception(
                "Expect 4 or 6 values for padding (padLeft, padRight, padTop, padBottom, [padFront, padBack])"
            )

    def forward(self, x):
        assert (
            x.shape[-1] >= self.padRight and x.shape[-1] >= self.padLeft
        ), "Padding size should be less than the corresponding input dimension for the azimuth axis"

        if self.padBack > 0 or self.padFront > 0:
            x = F.pad(x, (0, 0, 0, 0, self.padFront, self.padBack), "replicate")

        input_shape = x.shape
        x = x.view((x.shape[0], -1, x.shape[-2], x.shape[-1]))

        x = F.pad(
            x, (0, 0, self.padTop, self.padBottom), "reflect"
        )  # Actually, it should add a pi shift

        x = torch.cat((x[..., -self.padLeft :], x, x[..., : self.padRight]), dim=-1)

        return x.view((x.shape[0],) + input_shape[1:-2] + (x.shape[-2], x.shape[-1]))


class Mlp(nn.Module):
    """Multi-layer perceptron"""

    def __init__(self, in_features, out_features, hidden_features,
                 num_layers, activation="relu", dropout=0, batch_norm=False,
                 output_activation=None, dtype=torch.float32):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.dtype = dtype

        if num_layers == 1:
            self.blocks.append(
                self._create_mlp_block(in_features, out_features,
                                       batch_norm, activation, dropout)
            )
        else:
            for i in range(num_layers):
                if i == 0: # First layer
                    in_features_i = in_features
                    out_features_i = hidden_features
                    activation = activation
                elif i < num_layers - 1: # Hidden layers
                    in_features_i = hidden_features
                    out_features_i = hidden_features
                    activation = activation
                else: # Last layer
                    in_features_i = hidden_features
                    out_features_i = out_features
                    activation = output_activation

                self.blocks.append(
                    self._create_mlp_block(in_features_i, out_features_i,
                                            batch_norm, activation, dropout)
                )

    def _create_mlp_block(self, in_features, out_features, batch_norm, activation, dropout):
        layers = nn.ModuleList()
        layers.append(nn.Linear(in_features, out_features, dtype=self.dtype))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features, dtype=self.dtype))
        if activation is not None:
            if activation == "relu":
                activation = nn.ReLU()
            elif activation == "prelu":
                activation = nn.PReLU()
            elif isinstance(activation, nn.Module):
                pass
            layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        for block in self.blocks:
            for layer in block:
                if isinstance(layer, nn.BatchNorm1d):
                    if len(x.shape) == 3:
                        # BatchNorm1d expects the features in the second dimension
                        # while a linear layer expects features in the last dimension
                        x = layer(x.transpose(1, 2)).transpose(1, 2)
                    elif len(x.shape) == 2:
                        x = layer(x)
                    else:
                        raise Exception("Input should be 2D or 3D")
                else:
                    x = layer(x)

        return x
