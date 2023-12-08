import json
import torch

from datasets.locata_dataset import LocataDataset
from loss import OneSourceLoss
from models.srp import Srp
from trainers.one_source_tracker import OneSourceTracker
from trainers.cross_3d import Cross3dTrainer
from trainers.neural_srp_one_source import NeuralSrpOneSource
from utils import plot_estimated_doa_from_acoustic_scene


def main():
    # load params
    with open('params.json') as json_file:
        params = json.load(json_file)

    # Only for the output filenames, change it also in Network declaration cell
    model_name = params["model"]

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

    print("Analyzing LOCATA dataset")

    dataset_locata = LocataDataset(params["path_locata"],
                                   params["dataset"]["array_test"], params["fs"],
                                   dev=True, tasks=(1,3,5), win_size=params["win_size"],
                                   hop_rate=params["hop_rate"])

    global_loss = 0
    n_total_frames = 0
    nb_samples = len(dataset_locata)
    for i in range(nb_samples):
        mic_sig_batch, acoustic_scene_batch = dataset_locata.get_batch(i, i + 1)
        
        model_output, targets = trainer.predict_batch(
                    mic_sig_batch, acoustic_scene_batch, is_train=False)
        
        acoustic_scene_batch[0]["DOAw_pred"] = model_output["doa_sph"][0].detach().cpu().numpy()
        acoustic_scene_batch[0]["DOAw"] = targets["doa_sph"][0].detach().cpu().numpy()
        plot_estimated_doa_from_acoustic_scene(acoustic_scene_batch[0], output_path=f"images/{i}.png")

        n_frames = model_output["doa_sph"][0].shape[0]
        loss = trainer.loss(model_output, targets)["rms_deg"].item()
        print(f"{i}: {loss} ({n_frames} frames)")
        global_loss += loss*n_frames

        n_total_frames += n_frames
    global_loss /= n_total_frames

    print("Global loss:", global_loss)


if __name__ == "__main__":
    main()
