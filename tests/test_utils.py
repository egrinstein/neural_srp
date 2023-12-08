import matplotlib.pyplot as plt
import os

from ..utils import generate_polygon, n_quartets


def test_generate_hexagon():
    os.makedirs('tests/temp', exist_ok=True)

    points = generate_polygon(6, 1) # radius = 1
    assert len(points) == 6

    # Plot the hexagon
    plt.figure()

    plt.scatter(points[:, 0], points[:, 1])
    plt.axis('equal')

    plt.savefig('tests/temp/hexagon.png')


def test_n_quartets():
    assert n_quartets(4) == 15
    assert n_quartets(5) == 45
    assert n_quartets(6) == 105
