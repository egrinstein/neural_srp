# Contains routines for feature extraction
#

import math
from typing import Any
import numpy as np
import librosa


def n_choose_r(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n - r)


class GccExtractor:
    def __init__(self, params, gcc_mode="first"):
        """
        :param params: parameters dictionary
        """

        self._fs = params["fs"]
        # TODO: move hop rate and win size out of neural_srp
        self._hop_len_s = params["hop_rate"]*params["win_size"]/params["fs"]
        self._nb_bins = params["nb_gcc_bins"]
        self._gcc_mode = gcc_mode

        self._hop_len = int(self._fs * self._hop_len_s)
        self._win_len = 2 * self._hop_len
        self._nfft = _next_greater_power_of_2(self._win_len)

        self._eps = 1e-8

        # Max audio length in samples
        audio_max_len_samples = params["dataset"]["max_audio_len_s"] * self._fs
        self._max_feat_frames = int(
            np.ceil(audio_max_len_samples / float(self._hop_len))
        )

    def _spectrogram(self, audio_input):
        nb_ch = audio_input.shape[1]

        spectra = []
        for ch_cnt in range(nb_ch):
            stft_ch = librosa.core.stft(
                np.asfortranarray(audio_input[:, ch_cnt]),
                n_fft=self._nfft,
                hop_length=self._hop_len,
                win_length=self._win_len,
                window="hann",
            )
            spectra.append(stft_ch[:, : self._max_feat_frames])
        spectra = np.stack(spectra, axis=-1).transpose(1, 0, 2)

        return spectra

    def _get_gcc(self, linear_spectra):
        nb_frames, nb_stft_bins, nb_ch = linear_spectra.shape

        if self._gcc_mode == "first":  # Use first channel as reference
            n_output_channels = nb_ch - 1
        else:  # All combinations
            n_output_channels = n_choose_r(nb_ch, 2)

        gcc_feat = np.zeros((nb_frames, self._nb_bins, n_output_channels))
        mic_pair_idxs = []

        cnt = 0
        for m in range(nb_ch):
            for n in range(m + 1, nb_ch):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                 # Compute the cross-power spectrum
                # R /= np.abs(R) + self._eps  # PHAT weighting
                cc = np.fft.irfft(np.exp(1.0j * np.angle(R))) # Compute the GCC-PHAT
                cc = np.concatenate(
                    [  # Only keep central self._nb_bins
                        cc[:, -self._nb_bins // 2 :],
                        cc[:, : self._nb_bins // 2],
                    ],
                    axis=-1,
                )
                gcc_feat[:, :, cnt] = cc
                cnt += 1
                mic_pair_idxs.append([m, n])
            if self._gcc_mode == "first":
                break

        gcc_feat = gcc_feat.transpose((0, 2, 1))
        # gcc_feat.shape = (nb_frames, n_output_channels, self._nb_bins)
        
        return gcc_feat

    def forward(self, audio_in, labels=None):
        spect = self._spectrogram(audio_in)

        feat = self._get_gcc(spect)

        return feat, labels

    def __call__(self, audio_in, labels=None):
        return self.forward(audio_in, labels)


def _next_greater_power_of_2(x):
    return 2 ** (x - 1).bit_length()


class WindowTargets:
    """Windowing transform.
    Create it indicating the window length (K), the step between windows and an optional
    window shape indicated as a vector of length K or as a Numpy window function.

    TODO: Move this code to pytorch
    """

    def __init__(self, K, step):
        self.K = K
        self.step = step

    def __call__(self, x, acoustic_scene):
        N_mics = x.shape[1]
        N_dims = acoustic_scene["DOA"].shape[1]
        L = x.shape[0]
        N_w = np.floor(L / self.step - self.K / self.step + 1).astype(int)

        if self.K > L:
            raise Exception(
                f"The window size can not be larger than the signal length ({L})"
            )
        elif self.step > L:
            raise Exception(
                f"The window step can not be larger than the signal length ({L})"
            )

        DOAw = to_frames(acoustic_scene["DOA"], self.K, self.step)

        for i in np.flatnonzero(
            np.abs(np.diff(DOAw[..., 1], axis=1)).max(axis=1) > np.pi
        ):
            # Avoid jumping from -pi to pi in a window
            DOAw[i, DOAw[i, :, 1] < 0, 1] += 2 * np.pi
        DOAw = np.mean(DOAw, axis=1)
        DOAw[DOAw[:, 1] > np.pi, 1] -= 2 * np.pi
        acoustic_scene["DOAw"] = DOAw

        # Window the VAD if it exists
        if "vad" in acoustic_scene:
            acoustic_scene["vad"] = to_frames(acoustic_scene["vad"], self.K, self.step)

        # Timestamp for each window
        acoustic_scene["tw"] = (
            np.arange(0, (L - self.K), self.step) / acoustic_scene["fs"]
        )

        # Return the original signal
        return x, acoustic_scene


def to_frames(x, frame_size, hop_size):
    """Converts a signal to frames. The first dimension of the signal is the dimension which is framed.

    Args:
        x (np.ndarray): Input signal.
        frame_size (int): Number of frames.
        hop_size (int): Step between frames.
    Returns:
        np.ndarray: Framed signal of shape (... , n_frames, frame_size)
    """

    x_shape = x.shape
    n_signal = x_shape[0]

    n_frames = int(n_signal / hop_size - frame_size / hop_size)

    n_signal = n_frames * hop_size + frame_size
    # Truncate the signal to fit an integer number of frames
    x = x[:n_signal]

    out_shape = (n_frames, frame_size) + x_shape[1:]
    x_frames = np.zeros(out_shape, dtype=x.dtype)

    for i in range(n_frames):
        x_frames[i] = x[i * hop_size : i * hop_size + frame_size]

    return x_frames
