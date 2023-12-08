import numpy as np
import torch
import torch.nn as nn

from .signal_processing import GCC, Window, get_gcc_bins, compute_tau_max
from .mic_selection import select_pairs


class Srp(nn.Module):
    """Compute the SRP-PHAT maps from the GCCs taken as input.
    In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K), the
    desired resolution of the maps (resTheta and resPhi), the microphone positions relative to the center of the
    array (mic_pos) and the sampling frequency (fs).
    With normalize=True (default) each map is normalized to ethe range [-1,1] approximately
    """

    def __init__(
        self,
        frame_size,
        hop_rate,
        resTheta,
        resPhi,
        fs,
        c=343.0,
        normalize=False,
        thetaMax=torch.pi,
        mic_pos=None,
        gcc_transform="phat",
        gcc_tau_max=None,
        estimate_doa=False,
        mic_selection_mode="distinct_angles",
        peak_picking_mode="argmax"
    ):
        super().__init__()

        self.window = Window(frame_size, int(frame_size * hop_rate), window="hann")

        if gcc_transform == "phat":
            self.gcc = GCC(frame_size, transform=gcc_transform, tau_max=gcc_tau_max)
        elif isinstance(gcc_transform, nn.Module):
            self.gcc = gcc_transform

        self.frame_size = frame_size
        self.resTheta = resTheta
        self.resPhi = resPhi
        self.fs = float(fs)
        self.c = c
        self.normalize = normalize
        self.estimate_doa = estimate_doa
        self.peak_picking_mode = peak_picking_mode

        self.theta_range = [0, thetaMax]
        self.phi_range = [-torch.pi, torch.pi]

        self.theta = torch.linspace(self.theta_range[0], self.theta_range[1], resTheta)
        self.phi = torch.linspace(self.phi_range[0], self.phi_range[1], resPhi + 1)[:-1]

        self.grid = None
        self.mic_pos = mic_pos
        if mic_pos is not None:
            # Precompute the grid of tau0
            self.grid = self._init_grid(mic_pos)
        
        self.mic_selection_mode = mic_selection_mode

    def forward(self, x):
        x_signal = x["signal"]
        mic_pos = x["mic_pos"][0]
        grid = self.grid

        if grid is None:
            self.grid = grid = self._init_grid(mic_pos)
        
        gcc_tau_max = compute_tau_max(mic_pos, self.c, self.fs)

        # 0. Apply the window
        x_signal = self.window(x_signal)

        # 1. Compute the GCCs
        x_gcc = self.gcc(x_signal, tau_max=gcc_tau_max)
        N = x_gcc.shape[2]

        grid[grid < 0] += x_gcc.shape[-1]
        maps = torch.zeros(
            list(x_gcc.shape[0:-3]) + [self.resTheta, self.resPhi], device=x_gcc.device
        ).float()
        
        # 2. Compute the SRP maps
        # 2.1 Perform microphone selection
        # TODO: compute the microphone selection on initialization
        pair_idxs = select_pairs(mic_pos, mode=self.mic_selection_mode)

        # 2.2 Compute the SRP maps
        for pair in pair_idxs:
            maps += x_gcc[..., pair[0], pair[1], grid[pair[0], pair[1], :, :]]

        if self.normalize:
            maps -= torch.mean(torch.mean(maps, -1, keepdim=True), -2, keepdim=True)
            maps += 1e-12  # To avoid numerical issues
            maps /= torch.max(torch.max(maps, -1, keepdim=True)[0], -2, keepdim=True)[0]
        # else:
        #     maps /= len(pair_idxs)
        x["signal"] = maps

        if self.estimate_doa:
            x["doa_sph"] = self._estimate_doa(maps)

        return x
    
    def _estimate_doa(self, srp_map):
        # Estimate the DOA from the SRP map
        # srp_map: tensor of shape [batch_size, resTheta, resPhi]
        batch_size, nb_frames, resTheta, resPhi = srp_map.shape
        # Compute the maximum of the SRP map
        if self.peak_picking_mode == "argmax":
            maximums = srp_map.view(batch_size, nb_frames, -1).argmax(dim=-1)

            # Compute the DOA index between 0-1
            max_the = (maximums / resPhi).float() / resTheta
            max_phi = (maximums % resPhi).float() / resPhi
            # Convert to radians
            max_the = max_the * (self.theta_range[1] - self.theta_range[0]) + self.theta_range[0]
            max_phi = max_phi * (self.phi_range[1] - self.phi_range[0]) + self.phi_range[0]

            return torch.stack([max_the, max_phi], dim=-1)
        elif self.peak_picking_mode == "weighted_sum":
            # Compute the weighted sum
            # First, compute the weights
            weights = srp_map.view(batch_size, nb_frames, -1)
            weights -= torch.max(weights, dim=-1, keepdim=True)[0]
            weights = torch.softmax(weights, dim=-1)
            weights = weights.view(batch_size, nb_frames, resTheta, resPhi, 1)

            # Create the grid
            grid = torch.stack(torch.meshgrid(self.theta, self.phi), dim=-1)
            grid = grid.unsqueeze(0).unsqueeze(0).repeat(batch_size, nb_frames, 1, 1, 1)
            grid = grid.to(srp_map.device)
            # Compute the weighted sum
            return torch.sum(weights * grid, dim=(-2, -3))


    def _init_grid(self, mic_pos):
        # If the microphone positions are provided,
        # the candidate TDOAs can be precomputed
        N = mic_pos.shape[0]

        tdoas = torch.empty(
            (self.resTheta, self.resPhi, N, N),
            device=mic_pos.device,
        )  # Time differences, floats

        for k in range(N):
            for l in range(N):
                grid_kl = torch.stack(
                    [
                        torch.outer(torch.sin(self.theta), torch.cos(self.phi)),
                        torch.outer(torch.sin(self.theta), torch.sin(self.phi)),
                        torch.tile(torch.cos(self.theta), [self.resPhi, 1]).transpose(0, 1),
                    ],
                    dim=2,
                )
                mic_diff = (mic_pos[l, :] - mic_pos[k, :]).unsqueeze(0).unsqueeze(0)

                tdoas[:, :, k, l] = (grid_kl.to(mic_diff.device)*mic_diff).sum(dim=-1) / self.c

        tau = get_gcc_bins(self.frame_size, mic_pos.device) / float(self.fs)

        tau0 = torch.zeros_like(tdoas, dtype=torch.long, device=mic_pos.device)
        for k in range(N):
            for l in range(N):
                for i in range(self.resTheta):
                    for j in range(self.resPhi):
                        tau0[i, j, k, l] = int(
                            torch.argmin(torch.abs(tdoas[i, j, k, l] - tau))
                        )
        tau0[tau0 > self.frame_size // 2] -= self.frame_size
        tau0 = tau0.permute([2, 3, 0, 1])

        return tau0
