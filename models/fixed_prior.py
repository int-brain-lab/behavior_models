from models import model
import torch
import numpy as np
from models import utils as mut


unsqueeze = lambda x: torch.unsqueeze(torch.unsqueeze(x, 0), -1)


class fixed_prior(model.Model):
    """
    Model where the prior is based on an exponential estimation of the previous stimulus side
    """

    def __init__(
        self,
        path_to_results,
        session_uuids,
        mouse_name,
        actions,
        stimuli,
        stim_side,
        single_zeta=True,
        repetition_bias=False,
    ):
        name = (
            "fixed_prior"
            + "_with_repBias" * repetition_bias
            + "_single_zeta" * single_zeta
        )
        if single_zeta:
            nb_params, lb_params, ub_params = (
                4,
                np.array([0, 0, 0, 0]),
                np.array([1, 1, 0.5, 0.5]),
            )
            std_RW = np.array([0.05, 0.05, 0.01, 0.01])
        else:
            nb_params, lb_params, ub_params = (
                5,
                np.array([0, 0, 0, 0, 0]),
                np.array([1, 1, 1, 0.5, 0.5]),
            )
            std_RW = np.array([0.05, 0.05, 0.05, 0.01, 0.01])
        self.repetition_bias = repetition_bias
        self.single_zeta = single_zeta
        if repetition_bias:
            nb_params += 1
            lb_params, ub_params = np.append(lb_params, 0), np.append(ub_params, 0.5)
            std_RW = np.append(std_RW, 0.01)

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
        if not self.repetition_bias and not self.single_zeta:
            alpha, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(arr_params).T
        elif not self.repetition_bias:
            alpha, zeta, lapse_pos, lapse_neg = torch.tensor(arr_params).T
        elif self.single_zeta:
            alpha, zeta, lapse_pos, lapse_neg, rep_bias = torch.tensor(arr_params).T
        else:
            alpha, zeta_pos, zeta_neg, lapse_pos, lapse_neg, rep_bias = torch.tensor(
                arr_params
            ).T
        act, stim, side = torch.tensor(act), torch.tensor(stim), torch.tensor(side)

        if self.single_zeta:
            zeta_pos, zeta_neg = zeta, zeta

        alpha = mut.unsqueeze(alpha)
        lapses, zetas = mut.get_parameters(
            lapse_pos, lapse_neg, side, zeta_pos, zeta_neg
        )

        rep_bias = None if not self.repetition_bias else rep_bias
        logp_ch, pActions, prediction_error = mut.compute_logp_ch_and_pe(
            alpha, stim, act, side, zetas, lapses, rep_bias
        )

        if return_details:
            return logp_ch, alpha, pActions
        return np.array(torch.sum(logp_ch, axis=(0, -1)))
