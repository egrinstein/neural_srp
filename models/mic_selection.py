# Geometric microphone pair selection functionality:
# given a list of M microphone positions, select a subset of pairs
# from the available (M choose 2) pairs.
# The selection is based on the angle of each microphone pairs.
# The angle is defined as the angle between the line connecting the
# two microphones and the x-axis.
# The selection is done by comparing the cosine similarity of each pair vector
# Only the pairs with the biggest length are kept.

import torch

EPS = 1e-6


def select_pairs(mic_positions, mode="distinct_angles", threshold=0.05):
    device = mic_positions.device
    mic_positions = mic_positions.to("cpu").detach()
    
    if mode not in [
        "distinct_angles",
        "random",
        "first",
        "all"
    ]:
        raise NotImplementedError(f"""Mode {mode} not implemented.
                                  Choose from 'distinct_angles' 'random', 'first' or 'all'.""")
    
    # Generate all possible pairs
    M = len(mic_positions)
    pair_idxs = []
    for i in range(M):
        for j in range(i + 1, M):
            pair_idxs.append((i, j))

    pair_idxs = torch.Tensor(pair_idxs).long().to(device)
    P = len(pair_idxs)
    
    if mode == "all":
        return pair_idxs
    elif mode == "first":
        return pair_idxs[:M]

    # Else: 'distinct_angles' mode
    # Calculate the angles of each pair
    pair_vectors = []
    for pair in pair_idxs:
        pair_vector = mic_positions[pair[1]] - mic_positions[pair[0]]
        pair_vectors.append(pair_vector)

    pair_vectors = torch.stack(pair_vectors)
    # Calculate the angle distances between each quartet (pair of pairs)

    pair_idxs = tuple(tuple(sub) for sub in pair_idxs.tolist())
    out_pair_idxs = set(pair_idxs)
    pairs_to_discard = set()
    for i in range(P):
        if pair_idxs[i] in pairs_to_discard:
            continue
        same_angle_pairs = []
        lengths_same_angle_pairs = []
        for j in range(i + 1, P):
            if pair_idxs[j] in pairs_to_discard:
                continue
            # 1. compute the length distance between the two pairs
            length_i = torch.linalg.norm(pair_vectors[i])
            length_j = torch.linalg.norm(pair_vectors[j])
            
            # 2. compute the angle distance between the two pairs
            # using the cosine similarity
            angle_dist = torch.dot(pair_vectors[i], pair_vectors[j]) / (length_i * length_j)
            angle_dist = torch.abs(angle_dist)
            
            if angle_dist >= 1 - threshold:

                same_angle_pairs.append(pair_idxs[j])
                lengths_same_angle_pairs.append(length_j)

        same_angle_pairs.append(pair_idxs[i])
        lengths_same_angle_pairs.append(length_i)

        # Sort the pairs with the same angle by the length distance
        same_angle_pairs = torch.Tensor(same_angle_pairs)
        lengths_same_angle_pairs = torch.Tensor(lengths_same_angle_pairs)
        same_angle_pairs_sorted_idx = torch.argsort(lengths_same_angle_pairs)
        same_angle_pairs_sorted = same_angle_pairs[same_angle_pairs_sorted_idx]

        # Discard all pairs but the one with the biggest length
        for pair in same_angle_pairs_sorted[:-1]:
            pair = tuple(pair.tolist())
            pairs_to_discard.add(pair)

    # Remove the pairs to discard from the output pairs
    out_pair_idxs = out_pair_idxs - pairs_to_discard

    P_out = len(out_pair_idxs)
    # print(f"Selected {P_out} pairs out of {P} possible pairs.")

    if mode == "random":
        pair_idxs = torch.Tensor(pair_idxs).long().to(device)
        # Select the same number of pairs as the unique angles mode
        # for a fair comparison
        random_idxs = torch.randperm(P)
        return pair_idxs[random_idxs[:P_out]]

    return torch.Tensor(list(out_pair_idxs)).long().to(device)
