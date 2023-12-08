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
import sys
from models.nw_gcc import NwGCC
from models.srp import Srp
import torch

from datetime import datetime

from datasets.locata_dataset import LocataDataset
from loss import OneSourceLoss
from trainers.cross_3d import Cross3dTrainer
from trainers.neural_srp_one_source import NeuralSrpOneSource
from trainers.one_source_tracker import OneSourceTracker
from utils import Parameter


def _print_and_flush(msg):
	print(msg)
	sys.stdout.flush()


def main():
	# 1. load params
	with open('params.json') as json_file:
		params = json.load(json_file)
	print("Training parameters: ", params)

	T = params["dataset"]["max_audio_len_s"]
	max_rt60 = params["dataset"]["max_rt60"]
	min_snr = params["dataset"]["min_snr"]

	batch_size = params["training"]["batch_size"]
	lr = params["training"]["lr"]
	nb_epoch = params["training"]["nb_epochs"]
	nb_epoch_snr_decrease = params["training"]["nb_epoch_snr_decrease"]

	model_name = params["model"]  # Only for the output filenames, change it also in Network declaration cell

	# Load loss
	loss = OneSourceLoss(params)

	# %% Load network
	if model_name.startswith("neural_srp"):	
		trainer = NeuralSrpOneSource(params, loss)
	elif model_name == "cross_3d":
		trainer = Cross3dTrainer(params, loss, apply_vad=True)
	elif model_name == "srp":
		if params["srp"]["gcc_mode"] == "neural":
			gcc_transform = NwGCC(
				params["win_size"], transform="phat", tau_max=params["nb_gcc_bins"]//2
			)
		else:
			gcc_transform = "phat"
	
		model = Srp(params["win_size"], params["hop_rate"],
				params["srp"]["res_the"], params["srp"]["res_phi"],
				params["fs"], estimate_doa=True,
				mic_selection_mode=params["mic_pair_sampling_mode"],
				gcc_tau_max=params["nb_gcc_bins"] // 2,
				gcc_transform=gcc_transform,
				peak_picking_mode="weighted_sum")
		trainer = OneSourceTracker(model, loss)

	# 4. Load dataset
	if torch.cuda.is_available():
		trainer.cuda()
		
		# Avoid loading gpuRIR if not needed, so that the code can be tested on a machine without a GPU
		from datasets.librispeech_dataset import LibriSpeechDataset
		from datasets.random_trajectory_dataset import RandomTrajectoryDataset

		path_train = params["path_train"]
		path_test = params["path_test"]
		source_signal_dataset_train = LibriSpeechDataset(path_train, T, return_vad=True)
		source_signal_dataset_test = LibriSpeechDataset(path_test, T, return_vad=True)

		dataset_train = RandomTrajectoryDataset(
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
	else:
		print("No GPU available, using LOCATA dataset")
		dataset_train = LocataDataset(params["path_locata"], params["dataset"]["array_train"], params["fs"], dev=True,
									tasks=(1,3,5), win_size=params["win_size"], hop_rate=params["hop_rate"])
		dataset_test = LocataDataset(params["path_locata"], params["dataset"]["array_test"], params["fs"], dev=True,
									tasks=(1,3,5), win_size=params["win_size"], hop_rate=params["hop_rate"])

	# %% Network training

	print('Training network...')
	best_epoch = 0
	best_val_metric = float('inf')
	start_time_str = datetime.now().strftime('%m-%d_%Hh%Mm')
	run_dir = f'logs/{model_name}_{start_time_str}'
	os.makedirs(run_dir, exist_ok=True)
	# Save params
	with open(os.path.join(run_dir, 'params.json'), 'w') as json_file:
		json.dump(params, json_file, indent=4)

	for epoch_idx in range(1, nb_epoch + 1):
		_print_and_flush('\nEpoch {}/{}:'.format(epoch_idx, nb_epoch))
		if epoch_idx == nb_epoch_snr_decrease:
			print('\nDecreasing SNR')
			# SNR between min_snr dB and 30dB after the model has started to converge
			dataset_train.SNR = Parameter(min_snr, 30)
			dataset_test.SNR = Parameter(min_snr, 30)
		
		if epoch_idx in params["training"]["lr_decrease_epochs"]:
			# Decrease the learning rate
			print('\nDecreasing learning rate')
			lr /= params["training"]["lr_decrease_factor"]
		
		trainer.train_epoch(dataset_train,
		      				batch_size,
							lr=lr,
							epoch=epoch_idx
		)
		loss_test, rmsae_test = trainer.test_epoch(dataset_test, batch_size)
		_print_and_flush('Test loss: {:.4f}, Test rmsae: {:.2f}deg'.format(loss_test, rmsae_test))

		# Save best model
		if rmsae_test < best_val_metric:
			best_val_metric = rmsae_test
			_print_and_flush('New best model found at epoch {}, saving...'.format(epoch_idx))
			
			last_best_epoch = best_epoch
			best_epoch = epoch_idx

			best_model_path = f'{run_dir}/best_ep{best_epoch}.bin'
			trainer.save_checkpoint(best_model_path)
			if last_best_epoch > 0:
				last_best_model_path = f'{run_dir}/best_ep{last_best_epoch}.bin'
				os.remove(last_best_model_path)

	print('\nTraining finished\n')


	# %% Save model
	_print_and_flush('Saving model...')
	
	trainer.save_checkpoint(f'{run_dir}/last.bin')
	_print_and_flush('Model saved.\n')


if __name__ == '__main__':
	main()
