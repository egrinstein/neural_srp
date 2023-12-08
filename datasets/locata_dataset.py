import numpy as np
import os
import random

import scipy
import soundfile
import pandas
import warnings
import webrtcvad

from torch.utils.data import Dataset

from datasets.array_setup import ARRAY_SETUPS
from models.numpy_transforms import WindowTargets
from utils import cart2sph_np


class LocataDataset(Dataset):
    """Dataset with the LOCATA dataset recordings and its corresponding Acoustic Scenes.
    When you access to an element you get both the simulated signals in the microphones and a metadata dict.
    """

    def __init__(
        self,
        path,
        array,
        fs,
        tasks=(1, 3, 5),
        recording=None,
        dev=False,
        win_size=1024,
        hop_rate=0.25,
    ):
        """
        path: path to the root of the LOCATA dataset in your file system
        array: string with the desired array ('dummy', 'eigenmike', 'benchmark2' or 'dicit'),
                or list of strings with the desired aforementioned arrays
        fs: sampling frequency (you can use it to downsample the LOCATA recordings)
        tasks: LOCATA tasks to include in the dataset (only one-source tasks are supported)
        recording: recordings that you want to include in the dataset (only supported if you selected only one task)
        dev: True if the groundtruth source positions are available
        transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
        """
        
        assert (
            recording is None or len(tasks) == 1
        ), "Specific recordings can only be selected for dataset with only one task"
        for task in tasks:
            assert task in (1, 3, 5), "Invalid task " + str(task) + "."

        self.path = path
        self.dev = dev
        self.tasks = tasks
        self.transforms = [
            WindowTargets(
                win_size,
                int(win_size * hop_rate),
            ),
        ]
        self.fs = fs

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)

        if isinstance(array, str):
            # Fixed array mode
            self.arrays = [array]
            self.array = array
        elif isinstance(array, list):
            self.arrays = array
            self.array = array[0]

        self.directories = {}
        for task in tasks:
            task_path = os.path.join(path, "task" + str(task))
            for recording in os.listdir(task_path):
                if recording == ".DS_Store":
                    continue
                arrays_dir = os.listdir(os.path.join(task_path, recording))
                for array in self.arrays:
                    if array in arrays_dir:
                        if array not in self.directories:
                            self.directories[array] = []
                        self.directories[array].append(
                            os.path.join(task_path, recording, array))
        
        for array in self.arrays:
            self.directories[array].sort()

    def __len__(self):
        return len(self.directories[self.array])

    def __getitem__(self, idx):
        directory = self.directories[self.array][idx]
        mic_signals, fs = soundfile.read(
            os.path.join(directory, "audio_array_" + self.array + ".wav")
        )
        if fs > self.fs:
            mic_signals = scipy.signal.decimate(mic_signals, int(fs / self.fs), axis=0)
            new_fs = fs / int(fs / self.fs)
            if new_fs != self.fs:
                warnings.warn("The actual fs is {}Hz".format(new_fs))
            self.fs = new_fs
        elif fs < self.fs:
            raise Exception(
                "The sampling rate of the file ({}Hz) was lower than self.fs ({}Hz".format(
                    fs, self.fs
                )
            )

        # Remove initial silence
        start = np.argmax(mic_signals[:, 0] > mic_signals[:, 0].max() * 0.15)
        mic_signals = mic_signals[start:, :]
        t = (np.arange(len(mic_signals)) + start) / self.fs

        df = pandas.read_csv(
            os.path.join(directory, "position_array_" + self.array + ".txt"), sep="\t"
        )
        array_pos = np.stack((df["x"].values, df["y"].values, df["z"].values), axis=-1)

        array_rotation = np.zeros((array_pos.shape[0], 3, 3))
        for i in range(3):
            for j in range(3):
                array_rotation[:, i, j] = df["rotation_" + str(i + 1) + str(j + 1)]

        df = pandas.read_csv(os.path.join(directory, "required_time.txt"), sep="\t")
        required_time = (
            df["hour"].values * 3600 + df["minute"].values * 60 + df["second"].values
        )
        timestamps = required_time - required_time[0]

        if self.dev:
            sources_pos = []
            trajectories = []
            for file in os.listdir(directory):
                if file.startswith("audio_source") and file.endswith(".wav"):
                    source_signal, fs_src = soundfile.read(
                        os.path.join(directory, file)
                    )
                    if fs > self.fs:
                        source_signal = scipy.signal.decimate(
                            source_signal, int(fs_src / self.fs), axis=0
                        )
                    source_signal = source_signal[start : start + len(t)]
                if file.startswith("position_source"):
                    df = pandas.read_csv(os.path.join(directory, file), sep="\t")
                    source_pos = np.stack(
                        (df["x"].values, df["y"].values, df["z"].values), axis=-1
                    )
                    sources_pos.append(source_pos)
                    trajectories.append(
                        np.array(
                            [
                                np.interp(t, timestamps, source_pos[:, i])
                                for i in range(3)
                            ]
                        ).transpose()
                    )
            sources_pos = np.stack(sources_pos)
            trajectories = np.stack(trajectories)

            DOA_pts = np.zeros(sources_pos.shape[0:2] + (2,))
            DOA = np.zeros(trajectories.shape[0:2] + (2,))
            for s in range(sources_pos.shape[0]):
                source_pos_local = np.matmul(
                    np.expand_dims(sources_pos[s, ...] - array_pos, axis=1),
                    array_rotation,
                ).squeeze()
                DOA_pts[s, ...] = cart2sph_np(source_pos_local)[:, 1:3]
                DOA[s, ...] = np.array(
                    [np.interp(t, timestamps, DOA_pts[s, :, i]) for i in range(2)]
                ).transpose()
            DOA[DOA[..., 1] < -np.pi, 1] += 2 * np.pi
        else:
            sources_pos = None
            DOA = None
            source_signal = np.NaN * np.ones((len(mic_signals), 1))

        acoustic_scene = {
            "room_sz": np.NaN * np.ones((3, 1)),
            "T60":np.NaN,
            "beta": np.NaN * np.ones((6, 1)),
            "SNR": np.NaN,
            "array_setup": ARRAY_SETUPS[self.array],
            "mic_pos": np.matmul(
                array_rotation[0, ...],
                np.expand_dims(ARRAY_SETUPS[self.array]["mic_pos"], axis=-1),
            ).squeeze()
            + array_pos[
                0, :
            ],
            "source_signal": source_signal,
            "fs": self.fs,
            "t":t - start / self.fs,
            "traj_pts": sources_pos[0, ...],
            "timestamps": timestamps - start / self.fs,
            "trajectory": trajectories[0, ...],
            "DOA":DOA[0, ...],
        }

        vad = np.zeros_like(source_signal)
        vad_frame_len = int(10e-3 * self.fs)
        n_vad_frames = len(source_signal) // vad_frame_len
        for frame_idx in range(n_vad_frames):
            frame = source_signal[
                frame_idx * vad_frame_len : (frame_idx + 1) * vad_frame_len
            ]
            frame_bytes = (frame * 32767).astype("int16").tobytes()
            vad[
                frame_idx * vad_frame_len : (frame_idx + 1) * vad_frame_len
            ] = self.vad.is_speech(frame_bytes, int(self.fs))
        acoustic_scene["vad"] = vad

        if self.transforms is not None:
            for t in self.transforms:
                mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)
        return mic_signals, acoustic_scene

    def get_batch(self, idx1, idx2):
        mic_sig_batch = []
        acoustic_scene_batch = []
        for idx in range(idx1, idx2):
            mic_sig, acoustic_scene = self[idx]
            mic_sig_batch.append(mic_sig)
            acoustic_scene_batch.append(acoustic_scene)

        return self._collate_fn(mic_sig_batch, acoustic_scene_batch)

    def shuffle(self):
        random.shuffle(self.directories[self.array])

    def _collate_fn(self, mic_sig_batch, acoustic_scene_batch):
        """Collate function for the get_batch method.
        
        Args:
            mic_sig_batch (list): list of microphone signals (numpy arrays of shape (n_samples, n_mics)
                                                             or (n_frames, n_freq_bins, n_mics))
            acoustic_scene_batch (list): list of acoustic scenes

        Returns:
        """

        batch_size = len(mic_sig_batch)

        idx = np.argmax([sig.shape[0] for sig in mic_sig_batch])
        out_sig_shape = (batch_size,) + mic_sig_batch[idx].shape
        
        idx = np.argmax([scene["DOAw"].shape[0] for scene in acoustic_scene_batch])
        scene_doa_out_shape = acoustic_scene_batch[idx]["DOAw"].shape
        
        idx = np.argmax([scene["vad"].shape[0] for scene in acoustic_scene_batch])
        scene_vad_out_shape = acoustic_scene_batch[idx]["vad"].shape

        mic_sig_batch_out = np.zeros(out_sig_shape)
        for i in range(batch_size):
            mic_sig_batch_out[i, :mic_sig_batch[i].shape[0]] = mic_sig_batch[i]

            doaw = np.zeros(scene_doa_out_shape)
            vad = np.zeros(scene_vad_out_shape)
            nb_cur_frames = acoustic_scene_batch[i]["DOAw"].shape[0]
            doaw[:nb_cur_frames] = acoustic_scene_batch[i]["DOAw"]
            vad[:nb_cur_frames] = acoustic_scene_batch[i]["vad"]
            acoustic_scene_batch[i]["DOAw"] = doaw
            acoustic_scene_batch[i]["vad"] = vad

        return mic_sig_batch_out, np.stack(acoustic_scene_batch)

    def set_random_array(self):
        "Run this at the beginning of each epoch to randomly select an array setup"
        self.array = random.choice(self.arrays)
