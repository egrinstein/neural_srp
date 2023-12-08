import numpy as np
import torch
import itertools


def prepare_mic_pos(mic_pos, mic_pair_idxs, mode="mic_positions"):
    batch_size, n_pairs = mic_pair_idxs.shape[:2]

    out = torch.zeros(
        (batch_size, n_pairs, 2, 3), dtype=torch.float32, device=mic_pos.device
    )

    for i in range(batch_size):
        for j in range(n_pairs):
            out[i, j, 0] = mic_pos[i, mic_pair_idxs[i, j, 0]]
            out[i, j, 1] = mic_pos[i, mic_pair_idxs[i, j, 1]]
    
    if mode == "mic_positions":
        out = out.reshape(
            (batch_size, n_pairs, 6) # 6 = (2 pairs) * (3 coordinates)
        )*100 # m to cm
    elif mode == "mic_diff_vector":
        out = out[:, :, 0] - out[:, :, 1]
    elif mode == "norm_mic_diff_vector":
        # Compute the difference vector between the two mics
        mic_diff = out[:, :, 0] - out[:, :, 1]
        # Normalize the difference vector
        mic_diff_norm = torch.norm(mic_diff, dim=2, keepdim=True)
        mic_diff = mic_diff / mic_diff_norm
        # Concatenate the difference vector with its norm
        out = torch.cat(
            (mic_diff, mic_diff_norm), dim=2
        ).reshape((batch_size, n_pairs, 4)) # 4 = (3 coordinates) + (1 norm)

    return out


def get_all_pairs(n, device=None):
    "Get all pairs of indices from 0 to n-1"
    pairs = np.array(list(itertools.combinations(range(n), 2)))

    return torch.tensor(
        pairs,
        dtype=torch.long, device=device
    )


def create_spatial_positional_encoding(v, d, n=100):
    """Create a spatial positional encoding of the given vector v.
    The encoding is a matrix of size n x d, where n is a parameter
    controlling the frequency of the encoding, and d is the dimension
    of the output encoding.

    Parameters
    ----------
    v : torch.Tensor
        The vector to encode of shape (batch_size, length)
    d : int
        The dimension of the output encoding.
    n : int = 100
        The frequency hyperparameter.

    Returns
    -------
    torch.Tensor
        The spatial positional encoding of the given vector.
    """

    batch_size, length = v.shape[:2]

    # Compute the positional encoding
    pos_enc = torch.zeros(
        (batch_size, d), dtype=torch.float32, device=v.device
    )
    idxs = torch.arange(length, dtype=torch.float32, device=v.device)

    for k in range(d):
        if k % 2 == 0:
            func = torch.sin
        else:
            func = torch.cos
        
        enc = func(
            idxs*v / (n**(k/d))
        )
        pos_enc[:, k] = enc.mean(dim=1)

    return pos_enc
