import gpuRIR
from models.numpy_transforms import WindowTargets
import numpy as np
import pickle
import random
import tempfile

from torch.utils.data import Dataset

from utils import Parameter, acoustic_power, cart2sph_np

from datasets.array_setup import ARRAY_SETUPS, generate_random_array_setup

DEFAULT_RANDOM_MIC_CONFIG = {
    "radius_range_in_m": [0.05, 0.1],
    "n_mic_range": [4, 8],
    "min_dist_between_mics_in_m": 0.01,
    "mode": "spherical",
}


class RandomTrajectoryDataset(Dataset):
    """Dataset Acoustic Scenes with random trajectories.
    The length of the dataset is the length of the source signals dataset.
    When you access to an element you get both the simulated signals in the microphones and a metadata dictionary.
    """

    def __init__(
        self,
        sourceDataset,
        room_sz,
        T60,
        abs_weights,
        array,
        array_pos,
        SNR,
        nb_points,
        random_mic_config=DEFAULT_RANDOM_MIC_CONFIG,
        cache=False,
        noise_type="omni",
        win_size=1024,
        hop_rate=0.25,
    ):
        """
        sourceDataset: dataset with the source signals (such as LibriSpeechDataset)
        room_sz: Size of the rooms in meters
        T60: Reverberation time of the room in seconds
        abs_weights: Absorption coefficients rations of the walls
        array: Named tuple with the characteristics of the array
        array_pos: Position of the center of the array as a fraction of the room size
        SNR: Signal to Noise Ratio
        nb_points: Number of points to simulate along the trajectory
        transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
        random_mic_config: Configuration of the random microphone array
        cache: If True, the simulated signals are cached after the first epoch in temporary files to avoid simulating them again
        noise_type: "omni" or "directional" noise

        """

        self.sourceDataset = sourceDataset

        self.shuffled_idxs = np.arange(len(sourceDataset)) # Start with unshuffled indexes

        if isinstance(array, str):
            # Fixed array mode
            self.arrays = [array]
        elif isinstance(array, list):
            # Multi array mode: one is randomly selected at each batch
            self.arrays = array
        
        self.random_mic_config = random_mic_config
        self.set_random_array()

        self.room_sz = room_sz if type(room_sz) is Parameter else Parameter(room_sz)
        self.T60 = T60 if type(T60) is Parameter else Parameter(T60)
        self.abs_weights = (
            abs_weights if type(abs_weights) is Parameter else Parameter(abs_weights)
        )
        self.array_pos = (
            array_pos if type(array_pos) is Parameter else Parameter(array_pos)
        )

        self.SNR = SNR if type(SNR) is Parameter else Parameter(SNR)
        self.nb_points = nb_points
        self.fs = sourceDataset.fs

        self.transforms = [
            WindowTargets(
                win_size,
                int(win_size * hop_rate),
            ),
        ]
        self.cache = cache
        if self.cache:
            self.cached_paths = {}
        
        self.noise_type = noise_type

    def __len__(self):
        return len(self.sourceDataset)

    def __getitem__(self, idx):
        idx = self.shuffled_idxs[idx]

        # if idx < 0:
        #     idx = len(self) + idx

        if self.cache and idx in self.cached_paths:
            # Load from cache
            pickle_path = self.cached_paths[idx]
            with open(pickle_path, "rb") as f:
                mic_signals, acoustic_scene = pickle.load(f)
            
            return mic_signals, acoustic_scene
        else:
            acoustic_scene = self.get_random_scene(idx)
            mic_signals = simulate(acoustic_scene)

            if self.transforms is not None:
                for t in self.transforms:
                    mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

            if self.cache:
                # Save to cache
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    pickle_path = f.name
                    pickle.dump((mic_signals, acoustic_scene), f)
                    self.cached_paths[idx] = pickle_path

            return mic_signals, acoustic_scene

    def get_batch(self, idx1, idx2):
        self.set_random_array()

        mic_sig_batch = []
        acoustic_scene_batch = []
        for idx in range(idx1, idx2):
            mic_sig, acoustic_scene = self[idx]
            mic_sig_batch.append(mic_sig)
            acoustic_scene_batch.append(acoustic_scene)

        return np.stack(mic_sig_batch), np.stack(acoustic_scene_batch)

    def get_random_scene(self, idx):
        # Source signal
        source_signal, vad = self.sourceDataset[idx]

        # Room
        room_sz = self.room_sz.get_value()
        T60 = self.T60.get_value()
        abs_weights = self.abs_weights.get_value()
        beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights)

        # Microphones
        array_pos = self.array_pos.get_value() * room_sz
        mic_pos = array_pos + self.array_setup["mic_pos"]

        # Trajectory points
        src_pos_min = np.array([0.0, 0.0, 0.0])
        src_pos_max = room_sz.copy()
        
        if self.array_setup["array_type"] == "planar":
            # If array is planar, make a planar trajectory in the
            # same height as the array
            src_pos_min[2] = array_pos[2]
            src_pos_max[2] = array_pos[2]

        src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
        src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)

        Amax = np.min(
            np.stack(
                (
                    src_pos_ini - src_pos_min,
                    src_pos_max - src_pos_ini,
                    src_pos_end - src_pos_min,
                    src_pos_max - src_pos_end,
                )
            ),
            axis=0,
        )

        A = np.random.random(3) * np.minimum(
            Amax, 1
        )  # Oscilations with 1m as maximum in each axis
        w = (
            2 * np.pi / self.nb_points * np.random.random(3) * 2
        )  # Between 0 and 2 oscilations in each axis

        traj_pts = np.array(
            [
                np.linspace(i, j, self.nb_points)
                for i, j in zip(src_pos_ini, src_pos_end)
            ]
        ).transpose()
        traj_pts += A * np.sin(w * np.arange(self.nb_points)[:, np.newaxis])

        if np.random.random(1) < 0.25:
            traj_pts = np.ones((self.nb_points, 1)) * src_pos_ini

        # Interpolate trajectory points
        timestamps = (
            np.arange(self.nb_points) * len(source_signal) / self.fs / self.nb_points
        )
        t = np.arange(len(source_signal)) / self.fs
        trajectory = np.array(
            [np.interp(t, timestamps, traj_pts[:, i]) for i in range(3)]
        ).transpose()

        snr = self.SNR.get_value()
        acoustic_scene = {
            "room_sz": room_sz,
            "T60": T60,
            "beta": beta,
            "SNR": snr,
            "array_setup": self.array_setup,
            "mic_pos": mic_pos,
            "source_signal": source_signal,
            "fs": self.fs,
            "t": t,
            "traj_pts": traj_pts,
            "timestamps": timestamps,
            "trajectory": trajectory,
            "DOA": cart2sph_np(trajectory - array_pos)[:, 1:3],
            "source_vad": vad,
        }

        if self.noise_type == "directional":
            # Create a noise source signal at the desired SNR
            # Interferer position
            # create a candidate interferer position
            interferer_pos = self.array_pos.get_value() * room_sz
            
            # make sure it is not too close to the array
            while np.linalg.norm(interferer_pos - array_pos) < 0.3*np.linalg.norm(room_sz):
                interferer_pos = self.array_pos.get_value() * room_sz

            acoustic_scene["interferer_pos"] = interferer_pos

        return acoustic_scene

    def set_random_array(self):
        self.array = random.choice(self.arrays)

        if self.array == "random":
            self.array_setup = generate_random_array_setup(
                self.random_mic_config["radius_range_in_m"],
                self.random_mic_config["n_mic_range"],
                self.random_mic_config["min_dist_between_mics_in_m"],
                self.random_mic_config["mode"]
            )
        else:
            self.array_setup = ARRAY_SETUPS[self.array]
        
        self.N = self.array_setup["mic_pos"].shape[0]

    def shuffle(self):
        random.shuffle(self.shuffled_idxs)


def simulate(acoustic_scene):
    """Get the array recording using gpuRIR to perform the acoustic simulations."""
    if acoustic_scene["T60"] == 0:
        Tdiff = 0.1
        Tmax = 0.1
        nb_img = [1, 1, 1]
    else:
        Tdiff = gpuRIR.att2t_SabineEstimator(
            12, acoustic_scene["T60"]
        )  # Use ISM until the RIRs decay 12dB
        Tmax = gpuRIR.att2t_SabineEstimator(
            40, acoustic_scene["T60"]
        )  # Use diffuse model until the RIRs decay 40dB
        if acoustic_scene["T60"] < 0.15:
            Tdiff = Tmax  # Avoid issues with too short RIRs
        nb_img = gpuRIR.t2n(Tdiff, acoustic_scene["room_sz"])

    nb_mics = len(acoustic_scene["mic_pos"])
    nb_traj_pts = len(acoustic_scene["traj_pts"])
    nb_gpu_calls = min(
        int(
            np.ceil(
                acoustic_scene["fs"]
                * Tdiff
                * nb_mics
                * nb_traj_pts
                * np.prod(nb_img)
                / 1e9
            )
        ),
        nb_traj_pts,
    )
    traj_pts_batch = np.ceil(
        nb_traj_pts / nb_gpu_calls * np.arange(0, nb_gpu_calls + 1)
    ).astype(int)

    RIRs_list = [
        gpuRIR.simulateRIR(
            acoustic_scene["room_sz"],
            acoustic_scene["beta"],
            acoustic_scene["traj_pts"][traj_pts_batch[0] : traj_pts_batch[1], :],
            acoustic_scene["mic_pos"],
            nb_img,
            Tmax,
            acoustic_scene["fs"],
            Tdiff=Tdiff,
            # orV_rcv=acoustic_scene["array_setup"].mic_orV,
        )
    ]
    for i in range(1, nb_gpu_calls):
        RIRs_list += [
            gpuRIR.simulateRIR(
                acoustic_scene["room_sz"],
                acoustic_scene["beta"],
                acoustic_scene["traj_pts"][traj_pts_batch[i] : traj_pts_batch[i + 1], :],
                acoustic_scene["mic_pos"],
                nb_img,
                Tmax,
                acoustic_scene["fs"],
                Tdiff=Tdiff,
            )
        ]
    RIRs = np.concatenate(RIRs_list, axis=0)
    mic_signals = gpuRIR.simulateTrajectory(
        acoustic_scene["source_signal"],
        RIRs,
        timestamps=acoustic_scene["timestamps"],
        fs=acoustic_scene["fs"],
    )
    mic_signals = mic_signals[0 : len(acoustic_scene["t"]), :]

    dp_RIRs = gpuRIR.simulateRIR(
        acoustic_scene["room_sz"],
        acoustic_scene["beta"],
        acoustic_scene["traj_pts"],
        acoustic_scene["mic_pos"],
        [1, 1, 1],
        0.1,
        acoustic_scene["fs"],
    )

    dp_signals = gpuRIR.simulateTrajectory(
        acoustic_scene["source_signal"],
        dp_RIRs,
        timestamps=acoustic_scene["timestamps"],
        fs=acoustic_scene["fs"],
    )

    ac_pow = np.mean(
        [acoustic_power(dp_signals[:, i]) for i in range(dp_signals.shape[1])]
    )

    if "interferer_pos" in acoustic_scene:
        # Directional noise
        interf_RIRs = gpuRIR.simulateRIR(
            acoustic_scene["room_sz"],
            acoustic_scene["beta"],
            acoustic_scene["interferer_pos"][np.newaxis],
            acoustic_scene["mic_pos"],
            nb_img,
            Tmax,
            acoustic_scene["fs"],
            Tdiff=Tdiff
        )

        interf_signals = gpuRIR.simulateTrajectory(
            np.random.standard_normal(mic_signals.shape[0]), # Gaussian interferer
            interf_RIRs,
            timestamps=acoustic_scene["timestamps"],
            fs=acoustic_scene["fs"],
        )
        ac_pow_noise = np.mean([acoustic_power(interf_signals[:,i]) for i in range(interf_signals.shape[1])])
        interf_signals = np.sqrt(ac_pow/10**(acoustic_scene["SNR"]/10)) / np.sqrt(ac_pow_noise) * interf_signals
        
        mic_signals += interf_signals[0 : len(acoustic_scene["t"]), :]
    else:
        # Omnidirectional noise
        # Compute SNR based on the acoustic power of the direct path
        noise_signals = np.random.standard_normal(mic_signals.shape) # Gaussian sensor noise

        sensor_noise = np.sqrt(
            ac_pow / 10 ** (acoustic_scene["SNR"] / 10)
        ) * noise_signals
        mic_signals += sensor_noise

    # Apply the propagation delay to the VAD information if it exists
    if "source_vad" in acoustic_scene:
        vad = gpuRIR.simulateTrajectory(
            acoustic_scene["source_vad"],
            dp_RIRs,
            timestamps=acoustic_scene["timestamps"],
            fs=acoustic_scene["fs"],
        )
        acoustic_scene["vad"] = (
            vad[0 : len(acoustic_scene["t"]), :].mean(axis=1)
            > vad[0 : len(acoustic_scene["t"]), :].max() * 1e-3
        )

    return mic_signals
