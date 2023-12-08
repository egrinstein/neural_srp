import matplotlib.pyplot as plt
import torch
import os

from datasets.mic_pos_utils import create_spatial_positional_encoding



def test_create_spatial_positional_encoding():
    temp_dir = "tests/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    vec = torch.Tensor([
        [1, 1, 1],
        [2, 2, 2],
        [-1, -1, -1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    pos_enc = create_spatial_positional_encoding(
        vec, d=64, n=100
    ).numpy()

    for i in range(vec.shape[0]):
        # Plot the positional encoding
        plt.figure()
        plt.plot(pos_enc[i], label=vec[i])
        plt.legend()
        plt.savefig(f"{temp_dir}/pos_enc_{i}.png")
        plt.close()
