# Plot the model's localization error histogram 
# For different values of the RT60 and SNR on the simulated dataset

import json
from datasets.librispeech_dataset import LibriSpeechDataset
import torch

from datasets.random_trajectory_dataset import RandomTrajectoryDataset
from loss import OneSourceLoss
from models.srp import Srp
from trainers.one_source_tracker import OneSourceTracker
from trainers.cross_3d import Cross3dTrainer
from trainers.neural_srp_one_source import NeuralSrpOneSource
from utils import Parameter


def main():
    # load params
    with open('params.json') as json_file:
        params = json.load(json_file)

    # Only for the output filenames, change it also in Network declaration cell
    model_name = params["model"]
    params["batch_size"] = 1

    # Load loss
    loss = OneSourceLoss(params)

    if model_name == 'neural_srp':	
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

    if torch.cuda.is_available():
        trainer.cuda()

    print("Analyzing simulated dataset")

    T = params["dataset"]["max_audio_len_s"]
    path_test = params["path_test"]
    source_signal_dataset_train = LibriSpeechDataset(path_test, T, return_vad=True)
    
    max_rt60 = params["dataset"]["max_rt60"]
    min_snr = params["dataset"]["min_snr"]

    dataset_simulated = RandomTrajectoryDataset(
        sourceDataset = source_signal_dataset_train,
        room_sz = Parameter([3,3,2.5], [10,8,6]),  	# Random room sizes from 3x3x2.5 to 10x8x6 meters
        T60 = Parameter(0.2, max_rt60) if max_rt60 > 0 else 0, # Random reverberation times from 0.2 to max_rt60 seconds
        abs_weights = Parameter([0.5]*6, [1.0]*6),  # Random absorption weights ratios between walls
        array = params["dataset"]["array_train"],
        array_pos = Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5]), # Ensure a minimum separation between the array and the walls
        SNR = Parameter(30), 	# Start the simulation with a low level of omnidirectional noise
        nb_points = 156,		# Simulate 156 RIRs per trajectory (independent from the SRP-PHAT window length
        random_mic_config=params["dataset"]["random_mic_config"],
        cache=params["dataset"]["cache_random_traj_dataset"],
        noise_type=params["dataset"]["noise_type"],
        win_size=params["win_size"],
        hop_rate=params["hop_rate"]
    )

    global_loss = 0
    n_total_frames = 0
    nb_samples = len(dataset_simulated)
    for i in range(nb_samples):
        mic_sig_batch, acoustic_scene_batch = dataset_simulated.get_batch(i, i + 1)
        model_output, targets = trainer.predict_batch(
                    mic_sig_batch, acoustic_scene_batch, is_train=False)
        
        acoustic_scene_batch[0]["DOAw_pred"] = model_output["doa_sph"][0].detach().cpu().numpy()
        acoustic_scene_batch[0]["DOAw"] = targets["doa_sph"][0].detach().cpu().numpy()
        
        n_frames = model_output["doa_sph"][0].shape[0]
        loss = trainer.loss(model_output, targets)["rms_deg"].item()
        rt60 = acoustic_scene_batch[0]["T60"]
        snr = acoustic_scene_batch[0]["SNR"]
        print(f"{i}: {loss} - rt60={rt60} - snr={snr}")
        global_loss += loss*n_frames

        n_total_frames += n_frames
    global_loss /= n_total_frames

    print("Global loss:", global_loss)


if __name__ == "__main__":
    main()
