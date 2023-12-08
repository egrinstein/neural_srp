import numpy as np

from tqdm import trange

from datasets.array_setup import generate_random_array_setup


def test_generate_random_array_setup():
    radius_range_in_m = [0.05, 0.1]
    n_mic_range = [4, 10]
    min_dist_between_mics_in_m = 0.01
    for i in trange(10000):
        array_setup = generate_random_array_setup(radius_range_in_m,
                                                  n_mic_range, min_dist_between_mics_in_m)
        
        assert array_setup["array_type"] == "3D"
        assert array_setup["mic_pos"].dtype == np.float64

        assert np.all(np.linalg.norm(array_setup["mic_pos"], axis=1) <= n_mic_range[1])
        assert np.all(np.linalg.norm(array_setup["mic_pos"], axis=1) >= n_mic_range[0])
