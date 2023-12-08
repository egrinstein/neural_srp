import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from tqdm import tqdm

from models.doanet import DoaNet
from models.neural_srp import NeuralSrp
from datasets.tau_nigens_dataset import TauNigensDataLoader

from metrics import MultiSourceMetrics
from utils import dict_to_device, dict_to_float
from models.hnet import HNetGRU
from utils import get_params

N_MAX_SOURCES = 2


def cartesian_to_spherical_np(cart):
    """Numpy version of cartesian_to_spherical
    
    Args:
        cart: [x, y, z]
        
    Returns:
        [radius, elevation, azimuth]    
    """
    
    xy2 = cart[..., 0]**2 + cart[..., 1]**2
    sph = np.zeros_like(cart)
    sph[..., 0] = np.sqrt(xy2 + cart[..., 2]**2)
    sph[..., 1] = np.arctan2(cart[..., 2], np.sqrt(xy2)) # for elevation angle defined from Z-axis down

    sph[..., 2] = np.arctan2(cart[..., 1], cart[..., 0])
    return sph


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    params = get_params()

    print("\nLoading the best model and predicting results on the testing split")
    print("\tLoading testing dataset:")

    data_gen_test = TauNigensDataLoader(params=params, split=1)
   
    data_in, data_out = data_gen_test.get_data_sizes()
    dump_figures = True

    if params["model"] == "doanet":
        model = DoaNet(data_in, data_out, params["neural_srp"]).to(device)
    elif params["model"] == "neural_srp":
        model = NeuralSrp(params["nb_gcc_bins"], params["neural_srp"]).to(device)

    hnet_model = HNetGRU(max_len=2).to(device)
    hnet_model.load_state_dict(
        torch.load("hnet_model.h5", map_location=torch.device("cpu"))
    )
    for model_params in hnet_model.parameters():
        model_params.requires_grad = False

    doa_metric = MultiSourceMetrics(params)

    # Load checkpoint
    checkpoint_path = params["model_checkpoint_path"]
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model = model.to(device)
    if dump_figures:
        dump_folder = os.path.join(
            "images/tau", os.path.basename(checkpoint_path).split(".")[0]
        )
        os.makedirs(dump_folder, exist_ok=True)

    model.eval()
    with torch.no_grad():
        i = 0
        for data, target in tqdm(data_gen_test.get_batch(), total=len(data_gen_test)):
            data = dict_to_float(dict_to_device(data, device))

            output = model(data)
            doa_output = output["doa_cart"]
            activity_output = output["activity"]
            # loss_dict = loss(output, target, activity_out)

            doa_metric.partial_compute_metric(target, doa_output, activity_output)
            activity_target = target[..., -N_MAX_SOURCES:].bool()
            doa_target = target[..., : -N_MAX_SOURCES]

            # Only plot last batch
            batch_size, n_frames, n_classes = doa_target.shape
            doa_target = doa_target.reshape(
                batch_size, n_frames, 3, N_MAX_SOURCES
            ).transpose(-1, -2)

            activity_output = activity_output > 0.5
            
            if not params["save_figures"]:
                continue

            for i in range(batch_size):
                plot_estimated_doa(
                    doa_output[i],
                    doa_target[i],
                    duration=params["dataset"]["tau_nigens"]["duration_s"],
                    output_activity=activity_output[i].transpose(0, 1),
                    target_activity=activity_target[i].transpose(0, 1),
                    output_path=dump_folder + f"/{i}.pdf",
                    is_input_sph=False,
                )

    (le, _, _, lr, lp, lf) = doa_metric.get_results()
    metrics = {"loc_error": le, "recall": lr, "precision": lp, "F1": lf}
    print(metrics)


def plot_estimated_doa(
    predicted_doa,
    target_doa,
    duration=1,
    source_signal=None,
    target_activity=None,
    output_activity=None,
    output_path=None,
    is_input_sph=False,
):
    """Plots the DOA groundtruth and its estimation.
    The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
    """
    fig = plt.figure()

    # If source_signal is not None, plot it on top
    if source_signal is not None:
        gs = fig.add_gridspec(7, 1)
        axs = fig.add_subplot(gs[1:, 0]), fig.add_subplot(gs[0, 0])
        time_steps = np.linspace(0, duration, source_signal.shape[0])
        axs[1].plot(time_steps, source_signal)
        plt.xlim(time_steps[0], time_steps[-1])
        plt.tick_params(
            axis="both",
            which="both",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
        )
    else:
        axs = [fig.subplots(1, 1)]

    time_steps = np.linspace(0, duration, target_doa.shape[0])

    if not is_input_sph:
        # Convert target and predicted doa to polar
        target_doa = cartesian_to_spherical_np(target_doa)[..., 1:]
        predicted_doa = cartesian_to_spherical_np(predicted_doa)[..., 1:]

    target_doa = target_doa.transpose(1, 0, 2)
    predicted_doa = predicted_doa.transpose(1, 0, 2)

    for n_track in range(target_doa.shape[0]):
        
        # Filter out non-active instants
        if target_activity is not None:
            time_steps_target = time_steps[target_activity[n_track]]
            target_doa_track = target_doa[n_track][target_activity[n_track]]
        else:
            time_steps_target = time_steps
            target_doa_track = target_doa[n_track]

        if output_activity is not None:
            time_steps_output = time_steps[output_activity[n_track]]
            predicted_doa_track = predicted_doa[n_track][output_activity[n_track]]
        else:
            time_steps_output = time_steps
            predicted_doa_track = predicted_doa[n_track]

        colors = ["navy", "#83d44c"]
        curve_names = ["Target", "Predicted"]
        curve_types = ["Azimuth", "Elevation"]
        linestyles = ["-", "--"]

        
        # TODO: Add track colors
        for i in range(2):  # Azimuth and elevation
            for j, (t_steps, doa) in enumerate(
                zip([time_steps_target, time_steps_output], [target_doa_track, predicted_doa_track])
            ):
                label = None
                if n_track == 0:
                    label = curve_names[j] + " " + curve_types[i]
                    
                axs[0].plot(
                    t_steps, doa[..., i] * 180 / np.pi, color=colors[i], label=label,
                    linestyle=linestyles[j]
                )

    plt.gca().set_prop_cycle(None)

    axs[0].legend(loc="best")
    axs[0].set_xlabel("time [s]")
    axs[0].set_ylabel("DOA [º]")
    axs[0].yaxis.set_label_position("right")
    axs[0].set_xlim(time_steps[0], time_steps[-1])

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
