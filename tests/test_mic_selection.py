import matplotlib.pyplot as plt
import os

from models.mic_selection import select_pairs
from utils import generate_regular_polygon, plot_pairs


def test_select_pairs():
    for n_polygon in [4, 5, 6, 7, 8, 9, 10]:
        _test_select_pairs(n_polygon)


def _test_select_pairs(n_polygon):
    os.makedirs('tests/temp', exist_ok=True)
    fig, axs = plt.subplots(nrows=3)

    # Generate the hexagon
    points = generate_regular_polygon(n_polygon, 1)

    # Select all pairs
    pair_idxs = select_pairs(points, mode="all")
    # Select distinct pairs
    distinct_pair_idxs = select_pairs(points, mode="distinct_angles")
    # Select distinct pairs
    random_pair_idxs = select_pairs(points, mode="random")
    
    plot_pairs(points, pair_idxs, ax=axs[0])
    plot_pairs(points, distinct_pair_idxs, ax=axs[1])
    plot_pairs(points, random_pair_idxs, ax=axs[2])

    axs[0].set_title("All pairs")
    axs[1].set_title("Unique directions")
    axs[2].set_title("Random pairs")
    
    # Remove axes
    for i, ax in enumerate(axs):
        ax.axis('off')
    
    plt.tight_layout()

    plt.savefig(f'tests/temp/pairs_{n_polygon}.png')
    plt.close()
