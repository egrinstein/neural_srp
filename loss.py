import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import rms_angular_error_deg
from trainers.hnet import HNetGRU


class OneSourceLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.doa_loss_weight = params["loss"]["doa_loss_weight"]
        self.activity_loss_weight = params["loss"]["activity_loss_weight"]
        self.vad_weighted_loss = params["loss"]["vad_weighted_loss"]
        self.temporal_regularity_weight = params["loss"]["temporal_regularity_weight"]

        self.vad_loss_func = nn.BCEWithLogitsLoss(reduction="mean")

        if params["loss"]["norm_type"] == "l1":
            self.loss_func = nn.L1Loss(reduction="none")
        elif params["loss"]["norm_type"] == "l2":
            self.loss_func = nn.MSELoss(reduction="none")
        else:
            raise ValueError(
                f"Unknown loss norm_type: {params['loss']['norm_type']}"
            )
    
    def forward(self, model_output, target):

        doa_loss = self.loss_func(
            model_output["doa_cart"], target["doa_cart"]
        )

        # Only consider doa loss for frames where vad_true is 1
        if "activity" in target and self.vad_weighted_loss:
            doa_loss *= target["activity"]
        doa_loss = doa_loss.mean()

        output = {
            "loss": doa_loss*self.doa_loss_weight,
            "doa_loss": doa_loss
        }

        if  self.activity_loss_weight > 0 and \
            "activity" in model_output and "activity" in target:
            
            vad_loss = self.vad_loss_func(
                model_output["activity"], target["activity"]
            )

            output["vad_loss"] = vad_loss
            output["loss"] += vad_loss*self.activity_loss_weight

        if self.temporal_regularity_weight > 0:
            # Compute temporal regularity loss, 
            # which smoothens the predicted doa over time

            # Compute the difference between the doa of the current frame and the next frame
            # and take the mean over all frames
            temporal_regularity_loss = torch.mean(
                torch.linalg.norm(
                    model_output["doa_cart"][:, 1:, :] - model_output["doa_cart"][:, :-1, :],
                    dim=-1
                )
            )

            output["reg_loss"] = temporal_regularity_loss
            output["loss"] += temporal_regularity_loss*self.temporal_regularity_weight
        
        # Compute RMS angular error (not part of the loss)
        batch_size, n_labels, n_classes = target["doa_sph"].shape
        output["rms_deg"] = rms_angular_error_deg(
            target["doa_sph"].reshape(batch_size * n_labels, n_classes),
            model_output["doa_sph"].reshape(batch_size * n_labels, n_classes),
            target["activity"].reshape(batch_size * n_labels),
        )

        return output


class MultiSourceLoss(nn.Module):
    def __init__(self, params, device, hnet_model=True):
        super().__init__()
        self.params = params
        self.loss_params = params["loss"]
        self.device = device
        self.activity_loss = nn.BCEWithLogitsLoss()
        self.criterion = nn.MSELoss()

        if hnet_model:
            # load Hungarian network for data association, and freeze all layers.
            self.hnet_model = HNetGRU(max_len=2).to(device)
            self.hnet_model.load_state_dict(
                torch.load("hnet_model.h5", map_location=torch.device("cpu"))
            )
            for model_params in self.hnet_model.parameters():
                model_params.requires_grad = False

    def forward(self, output, target, activity_out=None):
        metrics = {}

        batch_size, seq_len, nb_tracks, _ = output.shape

        # Interpolate target to match output shape
        target = torch.nn.functional.interpolate(
            target.transpose(1, 2), size=output.shape[1]
        ).transpose(1, 2)
        # Transposing is needed because interpolate works on the last n - 2 dimensions
        # (the first ones are batch and channel)

        # The last dimension of target consists of the doas in the first unique_classes
        # and the activity vector in the last unique_classes
        target_doas = target[:, :, :-nb_tracks]
        target_activity = target[:, :, -nb_tracks:].reshape(-1, nb_tracks)
        nb_framewise_doas_gt = target_activity.sum(
            -1
        )  # Number of active doas in each frame

        # (batch, sequence, max_nb_doas*3) to (batch, sequence, 3, max_nb_doas)

        target_doas = target_doas.view(
            target_doas.shape[0], target_doas.shape[1], 3, nb_tracks
        ).transpose(-1, -2)

        # (batch, sequence, 3, max_nb_doas) to (batch*sequence, 3, max_nb_doas)

        # Collapse batch and sequence dimensions
        output = output.reshape(-1, output.shape[-2], output.shape[-1])
        target_doas = target_doas.reshape(
            -1, target_doas.shape[-2], target_doas.shape[-1]
        )

        # Compute unit-vectors of predicted DoA
        output_norm = torch.sqrt(torch.sum(output**2, -1) + 1e-10)
        output = output / output_norm.unsqueeze(-1)

        loss, metrics = self._compute_hnet_metrics(
            output, target_doas, nb_framewise_doas_gt, activity_out
        )

        metrics["loss"] = loss

        return metrics

    def _compute_hnet_metrics(
        self, output, target_doas, nb_framewise_doas_gt, activity_out
    ):
        metrics = {}

        # get pair-wise distance matrix between predicted and reference.
        dist_mat = torch.cdist(output.contiguous(), target_doas.contiguous())
        da_mat, _, _ = self.hnet_model(dist_mat)
        da_mat = da_mat.sigmoid()  # (batch*sequence, max_nb_doas, max_nb_doas)
        da_mat = da_mat.view(dist_mat.shape)
        da_mat = (da_mat > 0.5) * da_mat
        da_activity = da_mat.max(-1)[0]

        # Compute dMOTP loss for true positives
        dMOTP_loss = (
            torch.mul(dist_mat, da_mat).sum(-1).sum(-1)
            * da_mat.sum(-1).sum(-1)
            * self.params["loss"]["dMOTP_wt"]
        ).sum() / da_mat.sum()

        # Compute dMOTA loss
        M = da_activity.sum(-1)
        N = torch.Tensor(nb_framewise_doas_gt).to(self.device)
        FP = torch.clamp(M - N, min=0)
        FN = torch.clamp(N - M, min=0)
        IDS = (da_mat[1:] * (1 - da_mat[:-1])).sum(-1).sum(-1)
        IDS = torch.cat((torch.Tensor([0]).to(self.device), IDS))
        dMOTA_loss = (FP + FN + self.params["loss"]["IDS_wt"] * IDS).sum() / (
            M + torch.finfo(torch.float32).eps
        ).sum()

        metrics["dMOTP_loss"] = dMOTP_loss.item()
        metrics["dMOTA_loss"] = dMOTA_loss.item()

        loss = dMOTP_loss + self.params["loss"]["dMOTA_wt"] * dMOTA_loss

        if self.params["neural_srp"]["use_activity_output"]:
            activity_out = activity_out.view(-1, activity_out.shape[-1])
            act_loss = self.activity_loss(activity_out, (da_activity > 0.5).float())
            loss = loss + self.params["loss"]["activity_loss_weight"] * act_loss
            metrics["activity_loss"] = act_loss.item()

        return loss, metrics
