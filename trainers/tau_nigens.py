from datasets.tau_nigens_dataset import TauNigensDataLoader
import torch

import torch.optim as optim

from tqdm import tqdm

from models.doanet import DoaNet
from models.neural_srp import NeuralSrp, NeuralSrpFeatureExtractor
from utils import dict_to_device, dict_to_float, get_device


class TauNigensTrainer(torch.nn.Module):
    def __init__(self, params, loss, data_in, data_out, print_model=True,
                 allow_mps=True):
        super().__init__()
        self.device = get_device(allow_mps)

        self.params = params
        self.loss = loss

        # create model
        if params["model"] == "doanet":
            model = DoaNet(data_in, data_out, params["neural_srp"]).to(self.device)
        elif params["model"] == "neural_srp":
            model = NeuralSrp(params["nb_gcc_bins"], params["neural_srp"]).to(self.device)

        self.model = model
        
        # Only used during inference
        self.feature_extractor = NeuralSrpFeatureExtractor(params).to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=params["training"]["lr"])

        if params["model_checkpoint_path"] != "":
            self.load_checkpoint(params["model_checkpoint_path"])
        
        if print_model:
            print(model)

    def load_checkpoint(self, path):
        print(f"Loading model from checkpoint {path}")
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)

    def _epoch(self, data_generator: TauNigensDataLoader, mode, metric=None):
        nb_batches = 0
        loss_value = 0.0
        dMOTP_loss = 0.0
        dMOTA_loss = 0.0
        act_loss = 0.0

        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        for data, target in tqdm(data_generator.get_batch(), total=len(data_generator)):
            # load one batch of data

            data = dict_to_float(dict_to_device(data, self.device))
            target = target.to(self.device).float()
            # process the batch of data based on chosen mode

            if mode == "train":
                self.optimizer.zero_grad()
                output = self.model(data)
            else:
                with torch.no_grad():
                    output = self.model(data)

            activity_out = None
            if "activity" in output:
                activity_out = output["activity"]
            output = output["doa_cart"]

            loss_dict = self.loss(output, target, activity_out)

            if mode == "train":
                loss_dict["loss"].backward()
                self.optimizer.step()
            
            if metric is not None:
                metric.partial_compute_metric(target, output, activity_out)

            dMOTP_loss += loss_dict["dMOTP_loss"]
            dMOTA_loss += loss_dict["dMOTA_loss"]
            act_loss += loss_dict["activity_loss"]
            loss_value += loss_dict["loss"].item()

            nb_batches += 1
            if self.params["quick_test"] and nb_batches == 4:
                break

        loss_value /= nb_batches

        act_loss /= nb_batches
        dMOTP_loss /= nb_batches
        dMOTA_loss /= nb_batches

        return loss_value, dMOTP_loss, dMOTA_loss, act_loss 

    def train_epoch(self, data_generator):
        return self._epoch(data_generator, "train")

    def test_epoch(self, data_generator, metric=None):
        return self._epoch(data_generator, "test", metric)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.model(x)
