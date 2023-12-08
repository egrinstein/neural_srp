"""
    Trainer classes to train the models and perform inferences.

    File name: acousticTrackingTrainers.py
    Author: David Diaz-Guerra
    Date creation: 05/2020
    Python Version: 3.8
    Pytorch Version: 1.4.0
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

from utils import sph2cart, cart2sph
from datasets.locata_dataset import LocataDataset


class OneSourceTracker(torch.nn.Module):
    """Abstract class to the routines to train the one source tracking models and perform inferences."""

    def __init__(self, model, loss, checkpoint_path="", feature_extractor=None, print_model=False):
        super().__init__()

        self.model = model
        self.feature_extractor = feature_extractor

        self.cuda_activated = False
        self.loss = loss
        if checkpoint_path != "":
            self.load_checkpoint(checkpoint_path)
        
        if print_model:
            print(self.model)
            n_params = sum(p.numel() for p in self.model.parameters())
            print("Number of parameters: {}".format(n_params))

    def cuda(self):
        """Move the model to the GPU and perform the training and inference there."""
        self.model.cuda()
        self.cuda_activated = True

        if self.feature_extractor is not None:
            self.feature_extractor.cuda()
        
    def cpu(self):
        """Move the model back to the CPU and perform the training and inference here."""
        self.model.cpu()
        self.cuda_activated = False

    def forward(self, batch):
        if self.feature_extractor is not None:
            batch = self.feature_extractor(batch)
        return self.model(batch)

    def load_checkpoint(self, path):  
        print(f"Loading model from checkpoint {path}")
        state_dict = torch.load(path, map_location=torch.device('cpu'))        
        self.model.load_state_dict(state_dict)

    def save_checkpoint(self, path):
        print(f"Saving model to checkpoint {path}")
        torch.save(self.model.state_dict(), path)

    def extract_features(self, mic_sig_batch=None, acoustic_scene_batch=None, is_train=True):
        """Basic data transformation which
        moves the data to GPU tensors and applies the VAD.
        Override this method to apply more transformations."""

        output = {
            "network_input": {},
            "network_target": {},
        }

        # 1. Apply transform for mic signals
        if isinstance(mic_sig_batch, np.ndarray):
            mic_sig_batch = torch.from_numpy(mic_sig_batch.astype(np.float32))

        if self.cuda_activated:
            mic_sig_batch = mic_sig_batch.cuda()

        output["network_input"]["signal"] = mic_sig_batch

        # 2. Apply transform for acoustic scene
        if acoustic_scene_batch is not None:
            DOAw_batch = torch.from_numpy(
                np.stack(
                    [
                        acoustic_scene_batch[i]["DOAw"].astype(np.float32)
                        for i in range(len(acoustic_scene_batch))
                    ]
                )
            )

            mic_pos = torch.from_numpy(
                np.stack(
                    [
                        acoustic_scene_batch[i]["array_setup"]["mic_pos"].astype(
                            np.float32
                        )
                        for i in range(len(acoustic_scene_batch))
                    ]
                )
            )

            if self.cuda_activated:
                DOAw_batch = DOAw_batch.cuda()
                mic_pos = mic_pos.cuda()

            output["network_target"]["doa_sph"] = DOAw_batch
            output["network_input"]["mic_pos"] = mic_pos

            # 3. Apply transform for VAD
            if "vad" in acoustic_scene_batch[0]:
                vad_batch = np.stack(
                    [
                        acoustic_scene_batch[i]["vad"]
                        for i in range(len(acoustic_scene_batch))
                    ]
                )

                # Remove last dimension
                vad_batch = vad_batch.mean(axis=2) > 2 / 3

                vad_batch = torch.from_numpy(vad_batch)  # boolean
                if self.cuda_activated:
                    vad_batch = vad_batch.cuda()

                output["network_target"]["vad"] = vad_batch
            else:
                # Create a dummy, always on VAD
                doa_sph = output["network_target"]["doa_sph"]
                output["network_target"]["vad"] = torch.ones(doa_sph.shape[:2]).to(
                    doa_sph.device
                )

        if self.feature_extractor is not None:
            output["network_input"] = self.feature_extractor(
                output["network_input"]
            )

        return output

    def predict_batch(self, mic_sig_batch, acoustic_scene_batch=None, is_train=True):
        """Predict the DOA for a batch of trajectories."""
        data = self.extract_features(mic_sig_batch, acoustic_scene_batch, is_train=is_train)
        x_batch = data["network_input"]

        # if is_train:
        #     if isinstance(x_batch, dict):
        #         x_batch["signal"].requires_grad_()
        #         x_batch["mic_pos"].requires_grad_()
        #     else:
        #         x_batch = x_batch.requires_grad_()

        model_output = self.model(x_batch)

        # A model must output either a DOA in cartesian or spherical coordinates
        if "doa_sph" not in model_output:
            model_output["doa_sph"] = cart2sph(model_output["doa_cart"])
        if "doa_cart" not in model_output:
            model_output["doa_cart"] = sph2cart(model_output["doa_sph"])

        if acoustic_scene_batch is not None:
            # Prepare the targets for the loss function
            DOA_batch, vad_batch = self.interpolate_targets(
                data["network_target"], model_output["doa_cart"].shape)
            DOA_batch_cart = sph2cart(DOA_batch)

            targets = {
                "doa_cart": DOA_batch_cart,
                "doa_sph": DOA_batch,
                "activity": vad_batch
            }

            return model_output, targets
        else:
            return model_output

    def interpolate_targets(self, network_target, DOA_batch_pred_shape):
        """Interpolate the DOA and VAD targets to the same time instants as the predictions."""
        DOA_batch, vad_batch = network_target["doa_sph"], network_target["vad"]

        DOA_batch = F.interpolate(
            DOA_batch.transpose(-1, -2), size=DOA_batch_pred_shape[1], mode="linear"
        ).transpose(-1, -2)

        vad_batch = F.interpolate(
            vad_batch.unsqueeze(1).float(), size=DOA_batch_pred_shape[1], mode="linear"
        ).transpose(-1, -2)

        return DOA_batch, vad_batch

    def train_epoch(
        self,
        dataset,
        batch_size,
        lr=0.0001,
        shuffle=True,
        epoch=None
    ):
        """Train the model with an epoch of the dataset."""

        avg_vad_loss = 0
        avg_doa_loss = 0
        avg_reg_loss = 0
        avg_beta = 0.99

        self.model.train()  # set the model in "training mode"
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if shuffle:
            dataset.shuffle()
        
        n_trajectories = len(dataset)
        pbar = trange(n_trajectories // batch_size, ascii=True)
        for i in pbar:
            if epoch is not None:
                pbar.set_description("Epoch {}".format(epoch))

            optimizer.zero_grad()

            mic_sig_batch, acoustic_scene_batch = dataset.get_batch(
                i * batch_size,
                (i + 1) * batch_size,
            )
            model_output, targets = self.predict_batch(mic_sig_batch,
                                                       acoustic_scene_batch)

            loss = self.loss(model_output, targets)

            loss["loss"].backward()
            optimizer.step()
            
            avg_doa_loss = avg_beta * avg_doa_loss + (1 - avg_beta) * loss["doa_loss"].item()
            doa_loss = avg_doa_loss / (1 - avg_beta ** (i + 1))

            log_msg = {
                "doa_loss": doa_loss
            }

            if "vad_loss" in loss:
                avg_vad_loss = avg_beta * avg_vad_loss + (1 - avg_beta) * loss["vad_loss"].item()
                vad_loss = avg_vad_loss / (1 - avg_beta ** (i + 1))
                log_msg["vad_loss"] = vad_loss
            elif "reg_loss" in loss:
                avg_reg_loss = avg_beta * avg_reg_loss + (1 - avg_beta) * loss["reg_loss"].item()
                reg_loss = avg_reg_loss / (1 - avg_beta ** (i + 1))
                log_msg["reg_loss"] = reg_loss

            pbar.set_postfix(**log_msg)
            pbar.update()

            del model_output, targets, loss, mic_sig_batch, acoustic_scene_batch
        
        if isinstance(dataset, LocataDataset):
            dataset.set_random_array()

    def test_epoch(self, dataset, batch_size, sum_loss=True, return_labels=False):
        """Test the model with an epoch of the dataset."""
        self.model.eval()  # set the model in "testing mode"
        with torch.no_grad():
            loss_data = 0
            rmsae_data = 0

            if not sum_loss:
                loss_data = []
                rmsae_data = []
            
            if return_labels:
                labels = []

            n_trajectories = len(dataset)
            nb_batches = n_trajectories // batch_size
            
            for idx in trange(nb_batches):
                mic_sig_batch, acoustic_scene_batch = dataset.get_batch(
                    idx * batch_size, (idx + 1) * batch_size
                )
                
                model_output, targets = self.predict_batch(
                    mic_sig_batch, acoustic_scene_batch, is_train=False)

                loss = self.loss(model_output, targets)

                if sum_loss:
                    loss_data += loss["loss"].item()
                    rmsae_data += loss["rms_deg"]
                else:
                    loss_data.append(loss["loss"].cpu().item())
                    rmsae_data.append(loss["rms_deg"].cpu().item())
                
                if return_labels:
                    labels.append(acoustic_scene_batch)
 
            if sum_loss:
                loss_data /= nb_batches
                rmsae_data /= nb_batches
            else:
                loss_data = np.array(loss_data)
                rmsae_data = np.array(rmsae_data)

            out = loss_data, rmsae_data

            if return_labels:
                out = out + (labels,)
            
            return out
