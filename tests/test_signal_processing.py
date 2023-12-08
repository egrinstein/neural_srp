import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from models.signal_processing import GCC


def test_gcc():
    "Plot output of GCC function"
    temp_dir = "tests/temp"
    os.makedirs(temp_dir, exist_ok=True)

    gcc_func = GCC(4096, transform="phat", tau_max=40)

    # Create a Gaussian signal, pick subsignals 10 samples apart
    
    t = np.arange(4096)
    x = torch.normal(0, 1, size=(4096 + 10,))
    x1 = x[10:4096 + 10]
    x2 = x[:4096]

    x = torch.stack([x1, x2], dim=0).unsqueeze(0).unsqueeze(1)

    # Compute GCC
    gcc = gcc_func(x)
    gcc = gcc[0, 0, 0, 1].numpy() # GCC is a matrix, only take the first cross-element

    # Plot the two signals
    plt.figure()
    plt.plot(t, x1, label="x1")
    plt.plot(t, x2, label="x2")
    plt.legend()

    plt.savefig(f"{temp_dir}/gcc_signals.png")
    plt.close()

    # Plot the GCC
    plt.figure()
    plt.plot(gcc)
    plt.savefig(f"{temp_dir}/gcc.png")
    plt.close()