import torch
import torch.nn as nn
import typing

from torchaudio.transforms import MelScale


class Dft(nn.Module):
    """Compute the Mel-scaled phase spectra of the inputs."""

    def __init__(self, n_dft: int = 512, phase_only: bool = False):
        super().__init__()

        self.n_dft = _next_greater_power_of_2(n_dft*2)
        self.phase_only = phase_only

    def forward(self, x):
        batch_size, n_frames, n_mics, frame_size = x.shape
        x_fft = torch.fft.rfft(x, n=self.n_dft, dim=-1)    
        
        if self.phase_only:
            x_fft = torch.angle(x_fft)
        return x_fft


class MelSpectra(nn.Module):
    """Compute the Mel-scaled phase spectra of the inputs."""

    def __init__(self, sample_rate: int = 16000, n_dft: int = 512,
                 f_min: float = 0.0, f_max: typing.Optional[float] = None,
                 n_mels: int = 128, phase_only: bool = False):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_dft = _next_greater_power_of_2(n_dft)
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.phase_only = phase_only
 
        self.mel_scale_transform = MelScale(n_mels,
                                            sample_rate,
                                            f_min,
                                            f_max,
                                            n_stft=self.n_dft // 2 + 1)

    def forward(self, x):
        batch_size, n_frames, n_mics, frame_size = x.shape
        x_fft = torch.fft.rfft(x, n=self.n_dft, dim=-1)    
        x_fft = x_fft.permute(0, 2, 3, 1)
        # Permute to have freq, time as last two dimensions, as required by
        # torchaudio.transforms.MelScale
        
        x_mel_real = self.mel_scale_transform(
            x_fft.real
        )
        x_mel_imag = self.mel_scale_transform(
            x_fft.imag
        )
        x_mel = torch.complex(x_mel_real, x_mel_imag)

        # Permute back to original shape
        x_mel = x_mel.permute(0, 3, 1, 2)

        if self.phase_only:
            x_mel = torch.angle(x_mel)

        return x_mel


class MelCrossSpectra(MelSpectra):
    def __init__(self, sample_rate: int = 16000, n_dft: int = 512,
                 f_min: float = 0.0, f_max: typing.Optional[float] = None,
                 n_mels: int = 128, phase_only: bool = False):
        super().__init__(sample_rate, n_dft, f_min, f_max, n_mels, phase_only)
    
    def forward(self, x):
        # Compute the Mel-scaled phase spectra of the inputs.
        x_mel = super().forward(x)

        batch_size, n_frames, n_mics, n_mels = x_mel.shape

        # Compute the cross-spectra of the Mel-scaled phase spectra.
        cross_spectra = torch.zeros(
            batch_size, n_frames, n_mics, n_mics, n_mels,
            device=x.device, dtype=torch.complex64
        )

        for n in range(n_mics):
            cross_spectra[:, :, n] = x_mel[:, :, n].unsqueeze(2) * x_mel.conj()

        return cross_spectra


class GCC(nn.Module):
    """Compute the Generalized Cross Correlation of the inputs.
    In the constructor of the layer, you need to indicate the number of signals (N) and the window length (K).
    You can use tau_max to output only the central part of the GCCs and transform='PHAT' to use the PHAT transform.
    """

    def __init__(self, K, tau_max=None, transform=None, concat_bins=False, center=False, abs=True):
        assert (
            transform is None or transform == "phat"
        ), "Only the 'PHAT' transform is implemented"
        assert tau_max is None or tau_max <= K // 2
        super().__init__()

        self.K = K
        self.n_dft = _next_greater_power_of_2(K)
        self.tau_max = tau_max if tau_max is not None else K // 2
        self.transform = transform
        self.concat_bins = concat_bins
        self.center = center
        self.abs = abs

    def forward(self, x, tau_max=None):
        """Compute the GCCs of the input signals.
        x: tensor of shape [batch_size, n_frames, n_mics, frame_size] with K the number of time samples N the number of microphone channels
        tau_max (optional): maximum time lag in samples to output.

        Returns a tensor of shape [batch_size, N, N, 2*tau_max+1] with the GCCs of each pair of microphones
        """

        batch_size, n_frames, n_mics, frame_size = x.shape

        if tau_max is None:
            tau_max = self.tau_max

        x_fft = torch.fft.rfft(x, n=self.n_dft, dim=-1)
        if self.transform == "phat":
            x_fft /= x_fft.abs() + 1e-8  # To avoid numerical issues

        gcc = torch.empty(
            [batch_size, n_frames, n_mics, n_mics, 2 * tau_max],
            device=x.device,
        )
        for n in range(n_mics):
            gcc_fft_batch = x_fft[:, :, n].unsqueeze(2) * x_fft.conj()
            gcc_batch = torch.fft.irfft(gcc_fft_batch, dim=-1)

            if self.center:
                gcc[:, :, n, :, :tau_max] = gcc_batch[..., -tau_max:]
                gcc[:, :, n, :, tau_max:] = gcc_batch[..., :tau_max]
            else:
                gcc[:, :, n, :, :tau_max] = gcc_batch[..., :tau_max]
                gcc[:, :, n, :, -tau_max:] = gcc_batch[..., -tau_max:] 

        if self.concat_bins:
            # Return bins, in samples
            bins = get_gcc_bins(tau_max, x.device,
                                center=self.center)[:-1]
            # Remove last bin for consistency with the output shape
            bins = torch.ones_like(gcc)*bins
            gcc = torch.stack([gcc, bins], dim=-1)
        
        if self.abs:
            gcc = gcc.abs()

        return gcc

def _next_greater_power_of_2(x):
    return 2 ** (x - 1).bit_length()


class Window(nn.Module):
    """Windowing transform.
    Create it indicating the frame size, the step between windows and an optional
    window name indicating if it is rectangular or hanning.

    Returns a tensor of shape (batch_size, n_frames, K, n_mics)
    """

    def __init__(self, frame_size, hop_size, window=None):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window_name = window
        self.window = None

        super().__init__()

    def forward(self, x):
        batch_size, n_signal, n_mics = x.shape

        if self.frame_size > n_signal:
            raise Exception(
                f"The window size can not be larger than the signal length ({n_signal})"
            )
        elif self.hop_size > n_signal:
            raise Exception(
                f"The window step can not be larger than the signal length ({n_signal})"
            )
        
        if self.window is None:
            if self.window_name is None:
                self.window = torch.ones((self.frame_size, n_mics), dtype=x.dtype, device=x.device)
            elif self.window_name == "hann":
                self.window = torch.hann_window(self.frame_size, dtype=x.dtype, device=x.device)

        n_frames = int(n_signal / self.hop_size - self.frame_size / self.hop_size)
        n_signal = n_frames * self.hop_size + self.frame_size
        # Truncate the signal to fit an integer number of frames
        x = x[:, :n_signal]

        # Create a tensor with the shape (batch_size, n_frames, self.frame_size, n_mics)
        # where each frame is a window of the signal
        x_frames = torch.zeros(
            (batch_size, n_frames, self.frame_size, n_mics), dtype=x.dtype, device=x.device
        )

        for i in range(n_frames):
            x_frames[:, i] = x[:, i * self.hop_size : i * self.hop_size + self.frame_size] 

        x_frames =  x_frames*self.window.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        
        return x_frames.transpose(2, 3)


def get_gcc_bins(tau_max, device, center=True):
    if not center:
        raise NotImplementedError("Only center=True is implemented")
    # Return bins, in samples
    gcc_bins = torch.cat([
        torch.arange(0, tau_max, device=device),
        torch.arange(-tau_max, 0, device=device)
    ])

    gcc_bins = torch.zeros(2*tau_max + 1, device=device)
    gcc_bins[0:tau_max] = - torch.arange(tau_max, 0, -1)
    gcc_bins[tau_max:] = torch.arange(0, tau_max + 1)

    return gcc_bins


def compute_tau_max(mic_pos, c, fs):
    N = mic_pos.shape[0]

    dist_max = max(
        [
            max([
                torch.linalg.norm(
                    mic_pos[n, :] - mic_pos[m, :])
                for m in range(N)
            ])
            for n in range(N)
        ]
    )

    return int(torch.ceil(dist_max / c * fs))
