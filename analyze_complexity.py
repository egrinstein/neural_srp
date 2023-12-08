# Script that compares the computational complexity of the different
# localization algorithms (NeuralSRP, Cross3D and DOANet)
# It uses the "thop" library to compute the number of operations

import json
import matplotlib.pyplot as plt
import pandas as pd
import torch
import thop

from copy import deepcopy
from time import time

from trainers.cross_3d import Cross3dTrainer
from trainers.neural_srp_one_source import NeuralSrpOneSource
from trainers.tau_nigens import TauNigensTrainer
from trainers.one_source_tracker import OneSourceTracker
from models.srp import Srp

from datasets.array_setup import BENCHMARK2_ARRAY_SETUP

def main():
	# 1. load params
    with open('params.json') as json_file:
        params = json.load(json_file)
    print("Training parameters: ", params)

    params["model_checkpoint_path"] = ""

    params_doanet = deepcopy(params)
    params_doanet["model"] = "doanet"
    params_neural_srp_multi = deepcopy(params)
    params_neural_srp_multi["model"] = "neural_srp"

    n_mics = [4, 8, 12]

    results_mic_list = []
    for n_mic in n_mics:
        results_mic_list += analyze_models(
            n_mic, params_neural_srp_multi, params_doanet)

    results_mic_df = pd.DataFrame(results_mic_list)
    # Plot results
    # Create two stacked bar plots, one for the time and the other for the flops
    # for each model. The number of microphones is the stacked variable.
    
    # 5.1. time
    results_mic_df.pivot(index="model", columns="n_mics", values="time").plot.bar(
        rot=0, title="Time (ms) for different number of microphones", logy=True)

    plt.savefig("time.png")
    plt.clf()

    # 5.2. flops
    # Plot log in y axis
    results_mic_df.pivot(index="model", columns="n_mics", values="flops").plot.bar(
        rot=0, title="FLOPS (B) for different number of microphones", logy=True)
    
    plt.savefig("flops.png")

    results_mic_df.to_csv("results_mic.csv", index=False)


def analyze_models(n_mics, params, params_doanet):
    n_pairs = int(n_mics * (n_mics - 1) / 2)
    data_in = (1, n_pairs, 250, 64)
    data_out = (1, 50, 6)
    n_runs = 100

    # 2. create models
    models = {
        "Neural SRP": NeuralSrpOneSource(params, None),
        "Cross3D": Cross3dTrainer(params, None),
        "DOANet": TauNigensTrainer(params_doanet, None, data_in, data_out, allow_mps=False),
        "SRP": OneSourceTracker(
            Srp(params["win_size"], params["hop_rate"],
            params["srp"]["res_the"], params["srp"]["res_phi"],
            params["fs"], estimate_doa=True,
            mic_pos=torch.from_numpy(BENCHMARK2_ARRAY_SETUP["mic_pos"],),
            mic_selection_mode=params["mic_pair_sampling_mode"],
            gcc_tau_max=params["nb_gcc_bins"] // 2,
        ), None)
    }

    fs = params["fs"]
    # 3. create input: 1 second of noise
    signal = torch.randn(1, fs, n_mics)
    mic_pos = torch.randn(1, n_mics, 3)

    x_input = {
        "signal": signal,
        "mic_pos": mic_pos
    }

    results_list = []
    # 3. compute time
    for model_name, model in models.items():
        print("Model: ", model_name)
        avg_time = 0
        for i in range(n_runs):
            start = time()
            model(deepcopy(x_input))
            end = time()
            avg_time += (end - start)
        
        avg_time_ms = (avg_time * 1000) / n_runs
        print("Avg. time: %.2f ms" %(avg_time_ms))
        results_list.append(
            {
                "model": model_name,
                "time": avg_time_ms,
                "n_mics": n_mics
            }
        )
        print("-------")

    # 4. compute complexity
    for i, (model_name, model) in enumerate(models.items()):
        flops, params = thop.profile(model, inputs=(deepcopy(x_input), ))
        billion_flops = flops / 1e9
        million_params = params / 1e6
        print("Model: ", model_name)
        print("FLOPS: %.2f B" %(billion_flops))
        print("Params: %.2f M" %(million_params))
        print("-------")
        
        results_list[i]["flops"] = billion_flops
        results_list[i]["params"] = million_params

    return results_list


if __name__ == "__main__":
    main()
