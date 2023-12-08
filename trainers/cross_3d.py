"""
    Trainer classes to train the models and perform inferences.

    File name: acousticTrackingTrainers.py
    Author: David Diaz-Guerra
    Date creation: 05/2020
    Python Version: 3.8
    Pytorch Version: 1.4.0
"""

from models.cross_3d import Cross3D
import numpy as np
import torch
import webrtcvad

from models.srp import Srp
from models.nw_gcc import NwGCC

from trainers.one_source_tracker import OneSourceTracker
from datasets.array_setup import ARRAY_SETUPS


class Cross3dTrainer(OneSourceTracker):
    """Trainer for models which use SRP-PHAT maps as input"""

    def __init__(self, params, loss, apply_vad=False):
        """
        model: Model to work with
        N: Number of microphones in the array
        K: Window size for the SRP-PHAT map computation
        res_the: Resolution of the maps in the elevation axis
        res_phi: Resolution of the maps in the azimuth axis
        rn: Position of each microphone relative to te center of the array
        fs: Sampling frequency
        c: Speed of the sound [default: 343.0]
        array_type: 'planar' or '3D' whether all the microphones are in the same plane (and the maximum DOA elevation is pi/2) or not [default: 'planar']
        cat_maxCoor: Include to the network input tow addition channels with the normalized coordinates of each map maximum [default: False]
        apply_vad: Turn to zero all the map pixels in frames without speech signal [default: False]
        """

        cr_deep = int(min(4, np.log2(min(params["srp"]["res_the"], params["srp"]["res_phi"]))))
        # For low resolution maps it is not possible to perform 4 cross layers
        model = Cross3D(params["srp"]["res_the"], params["srp"]["res_phi"], cr_deep=cr_deep)

        super().__init__(model, loss,
                         checkpoint_path=params["model_checkpoint_path"],
                         feature_extractor=Cross3DFeatureExtractor(params))

        self.res_the = params["srp"]["res_the"]
        self.res_phi = params["srp"]["res_phi"]

        self.apply_vad = apply_vad
        if apply_vad:
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(3)

        self.array_train = params["dataset"]["array_train"]
        self.array_test = params["dataset"]["array_test"]

        self.mic_pos = None

    def activate_vad(self, apply=True):
        self.apply_vad = apply
        if self.apply_vad:
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(3)

    def extract_features(self, mic_sig_batch=None, acoustic_scene_batch=None, is_train=True):
        """Compute the SRP-PHAT maps from the microphone signals and extract the DoA groundtruth from the metadata dictionary."""

        output = super().extract_features(mic_sig_batch, acoustic_scene_batch)
        maps = output["network_input"]["signal"]

        if "vad" in output["network_target"]: # TODO: Move vad to network_input
            vad_output_th = output["network_target"]["vad"]
            vad_output_th = vad_output_th[:, np.newaxis, :, np.newaxis, np.newaxis]
            maps *= vad_output_th.float()

        output["network_input"]["signal"] = maps
        return output


class Cross3DFeatureExtractor(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        array_train = params["dataset"]["array_train"]
        array_test = params["dataset"]["array_test"]

        self.mic_pos = None
        if array_train == array_test \
            and array_train != "random":
            # If the train and test arrays are the same, the maximum delay is computed only once
            self.mic_pos = torch.from_numpy(ARRAY_SETUPS[array_train]["mic_pos"])


        win_size = params["win_size"]
        hop_rate = params["hop_rate"]

        self.c = params["speed_of_sound"]
        self.fs = params["fs"]

        self.res_phi = params["srp"]["res_phi"]

        gcc_mode = params["srp"]["gcc_mode"]
        if gcc_mode == "phat":
            gcc_transform = "phat"
        elif gcc_mode == "neural":
            gcc_transform = NwGCC(win_size, transform="phat", tau_max=params["nb_gcc_bins"]//2)

        self.srp = Srp(
            win_size,
            hop_rate,
            params["srp"]["res_the"],
            params["srp"]["res_phi"],
            self.fs,
            thetaMax=np.pi,
            mic_pos=self.mic_pos,
            gcc_transform=gcc_transform,
            mic_selection_mode=params["mic_pair_sampling_mode"],
            normalize=gcc_transform == "phat",
            # Only normalize if the GCC is not a neural network, because the gradient will be lost
        )
    
    def forward(self, x):
        x = self.srp(x)

        maps = x["signal"].unsqueeze(1)  # Add channel dimension
        maximums = maps.view(list(maps.shape[:-2]) + [-1]).argmax(dim=-1)

        max_the = (maximums / self.res_phi).float() / maps.shape[-2]
        max_phi = (maximums % self.res_phi).float() / maps.shape[-1]
        repeat_factor = np.array(maps.shape)
        repeat_factor[:-2] = 1
        repeat_factor = repeat_factor.tolist()
        maps = torch.cat(
            (
                maps,
                max_the[..., None, None].repeat(repeat_factor),
                max_phi[..., None, None].repeat(repeat_factor),
            ),
            1,
        )

        x["signal"] = maps
        return x
