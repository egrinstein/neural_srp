"""
    Python script to train the Cross3D model and analyze its performance.

    File name: 1sourceTracking_Cross3D.py
    Author: David Diaz-Guerra
    Date creation: 05/2020
    Python Version: 3.8
    Pytorch Version: 1.4.0
"""

import json
import os
import torch

from datetime import datetime

from datasets.librispeech_dataset import LibriSpeechDataset
from datasets.random_trajectory_dataset import RandomTrajectoryDataset
from models.srp import Srp
from loss import OneSourceLoss
from trainers.cross_3d import Cross3dTrainer
from trainers.neural_srp_one_source import NeuralSrpOneSource
from trainers.one_source_tracker import OneSourceTracker
from utils import Parameter


def main():
    # 1. load params
    with open('params.json') as json_file:
        params = json.load(json_file)
    print("Training parameters: ", params)

    T = params["dataset"]["max_audio_len_s"]
    max_rt60 = params["dataset"]["max_rt60"]
    min_snr = params["dataset"]["min_snr"]

    batch_size = params["training"]["batch_size"]

    model_name = params["model"]  # Only for the output filenames, change it also in Network declaration cell

    # Load loss
    loss = OneSourceLoss(params)

    # %% Load network
    if model_name.startswith("neural_srp"):	
        trainer = NeuralSrpOneSource(params, loss)
    elif model_name == "cross_3d":
        trainer = Cross3dTrainer(params, loss, apply_vad=True)
    elif model_name == "srp":
        model = Srp(params["win_size"], params["hop_rate"],
                    params["srp"]["res_the"], params["srp"]["res_phi"],
                    params["fs"], estimate_doa=True,
                    mic_selection_mode=params["mic_pair_sampling_mode"],
                    gcc_tau_max=params["nb_gcc_bins"] // 2)
        trainer = OneSourceTracker(model, loss)
        # SRP is not actually trained, but the OneSourceTracker is used
        # to prepare the data

    # 4. Load dataset
    if not torch.cuda.is_available():
        raise Exception("No GPU available, the simulations use gpuRIR which requires a GPU")

    trainer.cuda()
    
    path_test = params["path_test"]
    source_signal_dataset_test = LibriSpeechDataset(path_test, T, return_vad=True)

    dataset_test = RandomTrajectoryDataset(  # The same setup than for training but with other source signals
        sourceDataset = source_signal_dataset_test,
        room_sz = Parameter([3,3,2.5], [10,8,6]),
        T60 = Parameter(0.2, max_rt60) if max_rt60 > 0 else 0,
        abs_weights = Parameter([0.5]*6, [1.0]*6),
        array = params["dataset"]["array_test"],
        array_pos = Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5]),
        SNR = Parameter(min_snr, 30),
        nb_points = 156,
        random_mic_config=params["dataset"]["random_mic_config"],
        win_size=params["win_size"],
        hop_rate=params["hop_rate"]
    )
 
    start_time_str = datetime.now().strftime('%m-%d_%Hh%Mm')
    run_dir = f'logs/{model_name}_{start_time_str}'
    os.makedirs(run_dir, exist_ok=True)

    # 1. Test different reverb times with a fixed SNR
    print("Testing different reverb times with a fixed SNR")
    dataset_test.SNR = Parameter(20)
    rmsae_rt60, labels_rt60 = trainer.test_epoch(dataset_test, batch_size,
                                               sum_loss=False, return_labels=True)[1:]
    print("Mean rt60 error", rmsae_rt60.mean())
    # return rmsae_rt60, labels_rt60
    
    # 2. Test different SNRs with a fixed reverb time
    print("Testing different SNRs with a fixed reverb time")
    dataset_test.SNR = Parameter(min_snr, 30)
    dataset_test.T60 = Parameter(0.4, 0.4)
    rmsae_snr, labels_snr = trainer.test_epoch(dataset_test, batch_size,
                                               sum_loss=False, return_labels=True)[1:]
    
    print(rmsae_rt60, labels_rt60)
    print("Mean snr error", rmsae_snr.mean())
    
    return rmsae_rt60, labels_rt60, rmsae_snr, labels_snr


if __name__ == "__main__":
    main()