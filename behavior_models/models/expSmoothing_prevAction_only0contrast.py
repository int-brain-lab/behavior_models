from behavior_models.models import model
import torch
import numpy as np
from torch.distributions.normal import Normal
from behavior_models.models import utils as mut


class expSmoothing_prevAction_only0contrast(model.Model):
    """
    Model where the prior is based on an exponential estimation of the previous stimulus side
    """

    name = "actKernel"

    def __init__(
        self,
        path_to_results,
        session_uuids,
        mouse_name,
        actions,
        stimuli,
        stim_side,
        single_zeta,
    ):
        name = "actKernel_0contrast" + "_single_zeta" * single_zeta

        self.single_zeta = single_zeta
        nb_params = 4 + (not single_zeta) * 1
        lb_params, ub_params = np.zeros(nb_params), np.concatenate(
            (np.ones(nb_params - 2), np.array([0.5, 0.5]))
        )
        std_RW = np.concatenate(
            (
                np.array([0.05]),
                np.ones(1 + (not single_zeta) * 1) * 0.04,
                np.array([0.02, 0.02]),
            )
        )

        super().__init__(
            name,
            path_to_results,
            session_uuids,
            mouse_name,
            actions,
            stimuli,
            stim_side,
            nb_params,
            lb_params,
            ub_params,
            std_RW,
        )

    def compute_lkd(self, arr_params, act, stim, side, return_details):
        """
        Generates the loglikelihood (and prior)
        Params:
            arr_params (array): parameter of shape [nb_chains, nb_params]
            act (array of shape [nb_sessions, nb_trials]): action performed by the mice of shape
            stim (array of shape [nb_sessions, nb_trials]): stimulus contraste (between -1 and 1) observed by the mice
            side (array of shape [nb_sessions, nb_trials]): stimulus side (-1 (right), 1 (left)) observed by the mice
            return_details (boolean). If true, only return loglikelihood, else, return loglikelihood and prior
        Output:
            loglikelihood (array of length nb_chains): loglikelihood for each chain
            values (array of shape [nb_sessions, nb_chains, nb_trials, 2]): prior for each chain and session
        """
        nb_chains = len(arr_params)

        if self.single_zeta:
            alpha, zeta, lapse_pos, lapse_neg = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        else:
            alpha, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T

        act, stim, side = (
            torch.tensor(act, dtype=torch.float32),
            torch.tensor(stim, dtype=torch.float32),
            torch.tensor(side, dtype=torch.float32),
        )
        nb_sessions = len(act)

        values = (
            torch.zeros([nb_sessions, nb_chains, act.shape[-1], 2], dtype=torch.float32)
            + 0.5
        )

        if self.single_zeta:
            zeta_pos, zeta_neg = zeta, zeta

        alpha = mut.unsqueeze(alpha)
        lapses, zetas = mut.get_parameters(
            lapse_pos, lapse_neg, side, zeta_pos, zeta_neg
        )

        for t in range(act.shape[-1]):
            if t > 0:
                a_prev = torch.stack([act[:, t - 1] == -1, act[:, t - 1] == 1]) * 1
                values[:, :, t] = (1 - alpha) * values[:, :, t - 1] + alpha * torch.unsqueeze(a_prev.T, 1)
                not_to_update = (act[:, t - 1] == 0) + (stim[:, t - 1] != 0)
                values[not_to_update, :, t] = values[not_to_update, :, t - 1]

        logp_ch, pActions, prediction_error = mut.compute_logp_ch_and_pe(
            values[:, :, :, 1], stim, act, side, zetas, lapses, repetition_bias=None
        )

        if return_details:
            return logp_ch, values[:, :, :, 1], pActions
        return np.array(torch.sum(logp_ch, axis=(0, -1)))
