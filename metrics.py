import numpy as np
import torch

from scipy.optimize import linear_sum_assignment


class MultiSourceMetrics:
    def __init__(self, params):
        self.params = params

        self._eps = 1e-7

        self._localization_error = 0
        self._total_target = self._eps

        self._tp_doa = 0
        self._total_outputs = self._eps

        self._fp_doa = 0
        self._fn_doa = 0
        self._ids = 0

        return

    def _partial_compute_metric(self, dist_mat, target_activity, output_activity=None):
        M_max = dist_mat.shape[-1]
        
        N = target_activity.sum(-1)
        self._total_target += int(N.sum())

        if output_activity is not None:
            M = output_activity.sum(-1)
            self._fp_doa += (M - N).clip(min=0).sum(-1)
            self._fn_doa += (N - M).clip(min=0).sum(-1)

            self._ids += (output_activity[1:] * (1 - output_activity[:-1])).sum(-1).sum(-1)

        for frame_cnt, dist_mat_frame in enumerate(dist_mat):
            nb_active_target = int(N[frame_cnt])

            nb_active_output = M_max
            if output_activity is not None:
                nb_active_output = int(output_activity[frame_cnt].sum())
            
            self._total_outputs += nb_active_output

            # If there are true positives, compute the localization error
            if nb_active_target and nb_active_output:
                self._tp_doa += np.min((nb_active_target, nb_active_output))

                if output_activity is None:
                    if nb_active_target == 1:
                        dist_mat_frame = dist_mat_frame[:, 0][None]
                else:
                    # Keep only the active sources
                    dist_mat_frame = dist_mat_frame[output_activity[frame_cnt] == 1, :][
                        :, target_activity[frame_cnt] == 1
                    ]

                # linear_sum_assignment does the Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(dist_mat_frame)

                loc_err = dist_mat_frame[row_ind, col_ind].sum()

                self._localization_error += loc_err

    def get_results(self):
        # Localization error
        LE = self._localization_error / (self._tp_doa + self._eps)
        # Recall
        LR = self._tp_doa / (self._total_target + self._eps)
        # Precision
        LP = self._tp_doa / (self._total_outputs + self._eps)
        # F1-score
        LF = 2 * LP * LR / (LP + LR + self._eps)

        MOTa = 1 - (self._fp_doa + self._fn_doa + self._ids) / (
            self._total_target + self._eps
        )
        
        out = [
            180.0 * LE / np.pi,
            100.0 * MOTa,
            self._ids,
            100.0 * LR,
            100.0 * LP,
            100.0 * LF,
        ]

        return out

    def partial_compute_metric(self, target, output, activity_out):
        # Convert activity logits to probabilities
        activity_out = torch.sigmoid(activity_out)
        activity_out = activity_out.view(-1, activity_out.shape[-1])
        activity_binary = activity_out.cpu().detach().numpy() > 0.5

        max_nb_doas = output.shape[2]

        target_activity = target[:, :, -max_nb_doas:].reshape(-1, max_nb_doas)
        target_doas = target[..., :-max_nb_doas]

        target_doas = target_doas.view(
            target_doas.shape[0], target_doas.shape[1], 3, max_nb_doas
        ).transpose(-1, -2)

        # Compute unit-vectors of outputicted DoA
        # (batch, sequence, 3, max_nb_doas) to (batch*sequence, 3, max_nb_doas)
        output = output.view(-1, output.shape[-2], output.shape[-1])
        target_doas = target_doas.view(-1, target_doas.shape[-2], target_doas.shape[-1])

        # Noramlize the DoA vectors
        output_norm = torch.sqrt(torch.sum(output**2, -1, keepdim=True) + self._eps)
        output = output / output_norm

        # compute the angular distance matrix to estimate the localization error
        dot_prods = torch.matmul(output.detach(), target_doas.transpose(-1, -2))
        dot_prods = torch.clamp(dot_prods, -1 + self._eps, 1 - self._eps)
        # the +- eps is critical because the acos computation will become saturated if we have values of -1 and 1
        dist_mat_angle = torch.acos(dot_prods)

        self._partial_compute_metric(
            dist_mat_angle.cpu().detach().numpy(),
            target_activity.cpu().numpy(),
            output_activity=activity_binary,
        )


def angular_error(the_pred, phi_pred, the_true, phi_true):
    """ Angular distance between spherical coordinates.
    """
    aux = torch.cos(the_true) * torch.cos(the_pred) + \
          torch.sin(the_true) * torch.sin(the_pred) * torch.cos(phi_true - phi_pred)

    return torch.acos(torch.clamp(aux, -0.99999, 0.99999))


def mean_square_angular_error(y_pred, y_true):
    """ Mean square angular distance between spherical coordinates.
    Each row contains one point in format (elevation, azimuth).
    """
    the_true = y_true[..., 0]
    phi_true = y_true[..., 1]
    the_pred = y_pred[..., 0]
    phi_pred = y_pred[..., 1]

    return torch.pow(angular_error(the_pred, phi_pred, the_true, phi_true), 2)


def rms_angular_error_deg(y_pred, y_true, mask=None, mean=True):
    """ Root mean square angular distance between spherical coordinates.
    Each input row contains one point in format (elevation, azimuth) in radians
    but the output is in degrees.
    """

    if mask is not None:
        mask = mask.bool()
        y_pred = y_pred[mask]
        y_true = y_true[mask]

    error = torch.sqrt(mean_square_angular_error(y_pred, y_true)) * 180 / np.pi

    if mean:
        error = torch.mean(error)
    
    return error