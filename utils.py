"""
    Utils functions to deal with spherical coordinates in Pytorch.

    File name: utils.py
    Author: David Diaz-Guerra
    Date creation: 05/2020
    Python Version: 3.8
    Pytorch Version: 1.4.0
"""

import json
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import torch


def stack_dicts(dicts_list):
    """ Stacks a list of dictionaries into a single dictionary.
    """
    stacked_dict = {}
    for key in dicts_list[0].keys():
        stacked_dict[key] = np.stack([d[key] for d in dicts_list])
    return stacked_dict


def cart2sph(cart, include_r=False):
    """ Cartesian coordinates to spherical coordinates conversion.
    Each row contains one point in format (x, y, x) or (elevation, azimuth, radius),
    where the radius is optional according to the include_r argument.
    """
    r = torch.sqrt(torch.sum(torch.pow(cart, 2), dim=-1))
    theta = torch.acos(cart[..., 2] / r)
    phi = torch.atan2(cart[..., 1], cart[..., 0])
    if include_r:
        sph = torch.stack((theta, phi, r), dim=-1)
    else:
        sph = torch.stack((theta, phi), dim=-1)
    return sph


def cart2sph_np(cart, include_r=True):
    xy2 = cart[..., 0]**2 + cart[..., 1]**2
    sph = np.zeros_like(cart)
    sph[..., 0] = np.sqrt(xy2 + cart[..., 2]**2)
    sph[..., 1] = np.arctan2(np.sqrt(xy2), cart[..., 2]) # Elevation angle defined from Z-axis down
    sph[..., 2] = np.arctan2(cart[..., 1], cart[..., 0])
    
    if include_r:
        return sph
    else:
        return sph[..., 1:]

def sph2cart(sph):
    """ Spherical coordinates to cartesian coordinates conversion.
    Each row contains one point in format (elevation, azimuth, radius),
    where the radius is supposed to be 1 if it is not included.
    """
    if sph.shape[-1] == 2: sph = torch.cat((sph, torch.ones_like(sph[..., 0]).unsqueeze(-1)), dim=-1)
    x = sph[..., 2] * torch.sin(sph[..., 0]) * torch.cos(sph[..., 1])
    y = sph[..., 2] * torch.sin(sph[..., 0]) * torch.sin(sph[..., 1])
    z = sph[..., 2] * torch.cos(sph[..., 0])
    return torch.stack((x, y, z), dim=-1)


def acoustic_power(s):
    """ Acoustic power of after removing the silences.
    """
    w = 512  # Window size for silent detection
    o = 256  # Window step for silent detection

    # Window the input signal
    s = np.ascontiguousarray(s)
    sh = (s.size - w + 1, w)
    st = s.strides * 2
    S = np.lib.stride_tricks.as_strided(s, strides=st, shape=sh)[0::o]

    window_power = np.mean(S ** 2, axis=-1)
    th = 0.01 * window_power.max()  # Threshold for silent detection
    return np.mean(window_power[np.nonzero(window_power > th)])


class Parameter:
    """ Random parammeter class.
    You can indicate a constant value or a random range in its constructor and then
    get a value acording to that with get_value(). It works with both scalars and vectors.
    """
    def __init__(self, *args):
        if len(args) == 1:
            self.random = False
            self.value = np.array(args[0])
            self.min_value = None
            self.max_value = None
        elif len(args) == 2:
            self. random = True
            self.min_value = np.array(args[0])
            self.max_value = np.array(args[1])
            self.value = None
        else: 
            raise Exception('Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
    
    def get_value(self):
        if self.random:
            return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
        else:
            return self.value


def plot_pairs(points, pair_idxs, filename='', ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], label='# mics. = {}'.format(len(points)))
    ax.axis('equal')

    # Plot the pair vectors
    for i, pair_idx in enumerate(pair_idxs):
        label = None
        if i == 0:
            label = '# pairs = {}'.format(len(pair_idxs))
            
        mic_0 = points[pair_idx[0]]
        mic_1 = points[pair_idx[1]]
        ax.plot([mic_0[0], mic_1[0]], [mic_0[1], mic_1[1]], 'r', label=label)

    ax.legend()
    if filename:
        plt.savefig(filename)

    return ax


def plot_estimated_doa_from_acoustic_scene(acoustic_scene, output_path=None):
    """ Plots the DOA groundtruth and its estimation.
    The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
    """

    predicted_doa = acoustic_scene["DOAw_pred"]
    target_doa = acoustic_scene["DOAw"]
    vad = acoustic_scene["vad"]
    source_signal = acoustic_scene["source_signal"]
    duration = acoustic_scene["tw"][-1]

    plot_estimated_doa(predicted_doa, target_doa, duration, source_signal, vad, output_path)


def plot_estimated_doa(predicted_doa, target_doa, duration=1,
                          source_signal=None, vad=None, output_path=None):
    """ Plots the DOA groundtruth and its estimation.
    The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
    """
    fig = plt.figure()

    # If source_signal is not None, plot it on top
    if source_signal is not None:
        gs = fig.add_gridspec(7, 1)
        axs = fig.add_subplot(gs[1:,0]), fig.add_subplot(gs[0,0])
        time_steps = np.linspace(0, duration, source_signal.shape[0])
        axs[1].plot(time_steps, source_signal)
        plt.xlim(time_steps[0], time_steps[-1])
        plt.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
    else:
        axs = fig.subplots(1, 1)

    time_steps = np.linspace(0, duration, target_doa.shape[0])
    
    labels = ["Azimuth", "Elevation"]
    colors = ["navy", "#83d44c"]

    for i in range(target_doa.shape[1]):
        axs[0].plot(time_steps, target_doa[:, i] * 180/np.pi,
                      label=f"Target {labels[i]}", color=colors[i])
        axs[0].plot(time_steps, predicted_doa[:, i] * 180/np.pi, '--',
                      label=f"Predicted {labels[i]}", color=colors[i])

    plt.gca().set_prop_cycle(None)

    axs[0].legend(loc='best')
    axs[0].set_xlabel('time [s]')
    axs[0].set_ylabel('DOA [º]')
    axs[0].set_xlim(time_steps[0], time_steps[-1])
    axs[0].yaxis.set_label_position("right")

    # If vad is not None, plot it
    if vad is not None:
        silences = vad.mean(axis=1) < 2/3
        time_steps = np.linspace(0, duration, silences.shape[0])
        silences_idx = silences.nonzero()[0]
        start, end = [], []
        for i in silences_idx:
            if not i - 1 in silences_idx:
                start.append(i)
            if not i + 1 in silences_idx:
                end.append(i)
        for s, e in zip(start, end):
            axs[0].axvspan((s-0.5)*time_steps[1], (e+0.5)*time_steps[1], facecolor='0.5', alpha=0.5)

    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()


def dict_to_device(dict_of_tensors, device):
    """Move all the tensors in a dictionary to a device.
    Args:
        dict: A dictionary of tensors.
        device: The device to move the tensors to.
    """

    for key, value in dict_of_tensors.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
        elif isinstance(value, dict):
            value = dict_to_device(value, device)
        else:
            raise ValueError('Value is nor a tensor or a dictionary.')
        dict_of_tensors[key] = value

    return dict_of_tensors


def plot_pairs(points, pair_idxs, filename='', ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], label='# mics. = {}'.format(len(points)))
    ax.axis('equal')

    # Plot the pair vectors
    for i, pair_idx in enumerate(pair_idxs):
        label = None
        if i == 0:
            label = '# pairs = {}'.format(len(pair_idxs))
            
        mic_0 = points[pair_idx[0]]
        mic_1 = points[pair_idx[1]]
        ax.plot([mic_0[0], mic_1[0]], [mic_0[1], mic_1[1]], 'r', label=label)

    ax.legend()
    if filename:
        plt.savefig(filename)

    return ax


def dict_to_float(dict_of_tensors):
    """Convert all the tensors in a dictionary to float.
    Args:
        dict: A dictionary of tensors.
    """

    for key, value in dict_of_tensors.items():
        if isinstance(value, torch.Tensor):
            value = value.float()
        elif isinstance(value, dict):
            value = dict_to_float(value)
        dict_of_tensors[key] = value

    return dict_of_tensors


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)


def get_device(allow_mps=True):
    device = "cpu"
    # if torch.backends.mps.is_available() and allow_mps:
    #     device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return torch.device(device)


def get_params():
    # ########### default parameters ##############

    params = json.load(open("params.json", "r"))

    # Parameter manipulation
    tau_params = params["dataset"]["tau_nigens"]
    feature_label_resolution = int(tau_params["label_hop_len_s"] // tau_params["hop_len_s"])
    params["feature_sequence_length"] = (
        tau_params["label_sequence_length"] * feature_label_resolution
    )

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params


def generate_regular_polygon(n_sides, radius=1):
    """Generate a regular polygon with n_sides sides and radius radius."""

    points = []
    for i in range(n_sides):
        x = radius * math.cos(2 * math.pi * i / n_sides)
        y = radius * math.sin(2 * math.pi * i / n_sides)
        points.append([x, y])

    return torch.Tensor(points)
