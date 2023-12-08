import numpy as np

from utils import generate_regular_polygon


DICIT_ARRAY_SETUP = {
    'array_type': 'planar',
    'mic_pos': np.array([
        [ 0.96, 0.00, 0.00],
        [ 0.64, 0.00, 0.00],
        [ 0.32, 0.00, 0.00],
        [ 0.16, 0.00, 0.00],
        [ 0.08, 0.00, 0.00],
        [ 0.04, 0.00, 0.00],
        [ 0.00, 0.00, 0.00],
        [ 0.96, 0.00, 0.32],
        [-0.04, 0.00, 0.00],
        [-0.08, 0.00, 0.00],
        [-0.16, 0.00, 0.00],
        [-0.32, 0.00, 0.00],
        [-0.64, 0.00, 0.00],
        [-0.96, 0.00, 0.00],
        [-0.96, 0.00, 0.32]
    ]),
}

DUMMY_ARRAY_SETUP = {
    'array_type': 'planar',
    'mic_pos': np.array([
        [-0.079,  0.000, 0.000],
        [-0.079, -0.009, 0.000],
        [ 0.079,  0.000, 0.000],
        [ 0.079, -0.009, 0.000]
    ]), 
}

BENCHMARK2_ARRAY_SETUP = {
    'array_type': '3D',
    'mic_pos': np.array([
        [-0.028,  0.030, -0.040],
        [ 0.006,  0.057,  0.000],
        [ 0.022,  0.022, -0.046],
        [-0.055, -0.024, -0.025],
        [-0.031,  0.023,  0.042],
        [-0.032,  0.011,  0.046],
        [-0.025, -0.003,  0.051],
        [-0.036, -0.027,  0.038],
        [-0.035, -0.043,  0.025],
        [ 0.029, -0.048, -0.012],
        [ 0.034, -0.030,  0.037],
        [ 0.035,  0.025,  0.039]
    ]),
}

EIGENMIKE_ARRAY_SETUP = {
    'array_type': '3D',
    'mic_pos': np.array([
        [ 0.000,  0.039,  0.015],
        [-0.022,  0.036,  0.000],
        [ 0.000,  0.039, -0.015],
        [ 0.022,  0.036,  0.000],
        [ 0.000,  0.022,  0.036],
        [-0.024,  0.024,  0.024],
        [-0.039,  0.015,  0.000],
        [-0.024,  0.024,  0.024],
        [ 0.000,  0.022, -0.036],
        [ 0.024,  0.024, -0.024],
        [ 0.039,  0.015,  0.000],
        [ 0.024,  0.024,  0.024],
        [-0.015,  0.000,  0.039],
        [-0.036,  0.000,  0.022],
        [-0.036,  0.000, -0.022],
        [-0.015,  0.000, -0.039],
        [ 0.000, -0.039,  0.015],
        [ 0.022, -0.036,  0.000],
        [ 0.000, -0.039, -0.015],
        [-0.022, -0.036,  0.000],
        [ 0.000, -0.022,  0.036],
        [ 0.024, -0.024,  0.024],
        [ 0.039, -0.015,  0.000],
        [ 0.024, -0.024, -0.024],
        [ 0.000, -0.022, -0.036],
        [-0.024, -0.024, -0.024],
        [-0.039, -0.015,  0.000],
        [-0.024, -0.024,  0.024],
        [ 0.015,  0.000,  0.039],
        [ 0.036,  0.000,  0.022],
        [ 0.036,  0.000, -0.022],
        [ 0.015,  0.000, -0.039]
    ]), 
}

MINIDSP_ARRAY_SETUP = {
    'array_type': 'planar',
    'mic_pos': np.array([
        [ 0.0000,  0.0430, 0.000],
        [ 0.0372,  0.0215, 0.000],
        [ 0.0372, -0.0215, 0.000],
        [ 0.0000, -0.0430, 0.000],
        [-0.0372, -0.0215, 0.000],
        [-0.0372,  0.0215, 0.000]
    ]),
}

TAU_NIGENS_TETRAHEDRAL = {
    'array_type': '3D',
    'mic_pos': np.array([
        [ 0.0243,  0.0243,  0.024],
        [ 0.0243, -0.0243, -0.024],
        [-0.0243,  0.0243, -0.024],
        [-0.0243, -0.0243,  0.024]]),
}

ARRAY_SETUPS = {
    "benchmark2": BENCHMARK2_ARRAY_SETUP,
    "dicit": DICIT_ARRAY_SETUP,
    "dummy": DUMMY_ARRAY_SETUP,
    "eigenmike": EIGENMIKE_ARRAY_SETUP,
    "mini_dsp": MINIDSP_ARRAY_SETUP,
    "tau_nigens_tetrahedral": TAU_NIGENS_TETRAHEDRAL
}


def generate_random_array_setup(radius_range_in_m, n_mics_range,
                                min_dist_between_mics_in_m=0,
                                mode="spherical"):
    """Generate a random array setup with n_mics microphones
    on a sphere of radius between min_radius_in_m and max_radius_in_m meters
    
    Args:
        radius_range_in_m (tuple of floats): (min_radius_in_m, max_radius_in_m)
        n_mics_range (tuple of ints): (min_n_mics, max_n_mics)
        min_dist_between_mics_in_m (float, optional): Minimum distance between microphones in meters. Defaults to 0.
        mode (str, optional): Mode of the array, either "spherical" or "poly2d". "spherical" scatter points
         within a sphere of radius which is randomly assigned. "poly2d" generates a regular 2d polygon of n_mics vertices.
         Defaults to "spherical".
    Returns:
        dict: Dictionary containing the keys 'array_type' and 'mic_pos'
    """
    array_type = "3D"
    if mode == "poly2d":
        array_type = "planar"
    
    # 1. Generate a random number of microphones
    n_mics = np.random.randint(
        n_mics_range[0], n_mics_range[1]
    )

    array_setup = {
        'array_type': array_type,
        'mic_pos': np.zeros((n_mics, 3))
    }
    
    # 2. Generate a random radius
    radius_in_m = np.random.uniform(radius_range_in_m[0],
                                    radius_range_in_m[1])

    if mode == "spherical":
        # 3. Generate random points on the sphere
        array_setup['mic_pos'] = place_random_points_on_sphere(
            n_mics, radius_in_m, min_dist_between_mics_in_m)
    elif mode == "poly2d":
        # 3. Generate a regular polygon with n_mics vertices.
        # Note: min_dist_between_mics_in_m is not used here.
        array_setup['mic_pos'] = generate_regular_polygon(
            n_mics, radius_in_m)

    return array_setup


def place_random_points_on_sphere(n_points, radius_in_m, min_dist_between_points_in_m=0):
    points = np.zeros((n_points, 3))

    for i in range(n_points):
        while True:
            # 1. Generate a random angle inside the sphere
            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.random.uniform(0, np.pi)
            
            # 2. Convert angle to Cartesian coordinates
            x = np.cos(phi) * np.sin(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(theta)

            points[i] = radius_in_m*np.array([x, y, z])
            
            if i == 0:
                break
            
            # 3. If candidate point within distance from other points, keep it.
            dists = np.linalg.norm(points[:i] - points[i], axis=1)
            if np.all(dists >= min_dist_between_points_in_m):
                break
    
    return points
