from models import model, utils
import torch
import numpy as np
from models import utils as mut


class optimal_Bayesian(model.Model):
    """
    Model where the prior is based on an exponential estimation of the previous stimulus side
    """

    name = "optBay"

    def __init__(
        self,
        path_to_results,
        session_uuids,
        mouse_name,
        actions,
        stimuli,
        stim_side,
        single_zeta,
        repetition_bias=False,
    ):
        name = (
            "optimal_bayesian"
            + "_with_repBias" * repetition_bias
            + "_single_zeta" * single_zeta
        )
        self.single_zeta = single_zeta
        nb_params = 3 + (not single_zeta) * 1
        lb_params, ub_params = np.zeros(nb_params), np.concatenate(
            (np.ones(nb_params - 2), np.array([0.5, 0.5]))
        )
        std_RW = np.concatenate(
            (np.ones(1 + (not single_zeta) * 1) * 0.04, np.array([0.02, 0.02]))
        )
        self.repetition_bias = repetition_bias
        if repetition_bias:
            lb_params, ub_params = np.append(lb_params, 0), np.append(ub_params, 0.5)
            std_RW = np.append(std_RW, 0.01)
            nb_params += 1

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
        self.nb_blocklengths, self.nb_typeblocks = 100, 3

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
            prior (array of shape [nb_sessions, nb_chains, nb_trials]): prior for each chain and session
        """
        nb_chains = len(arr_params)
        if not self.repetition_bias and self.single_zeta:
            zeta, lapse_pos, lapse_neg = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        elif self.single_zeta:
            zeta, lapse_pos, lapse_neg, rep_bias = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        elif not self.repetition_bias:
            zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        else:
            zeta_pos, zeta_neg, lapse_pos, lapse_neg, rep_bias = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        act, stim, side = (
            torch.tensor(act, device=self.device, dtype=torch.float32),
            torch.tensor(stim, device=self.device, dtype=torch.float32),
            torch.tensor(side, device=self.device, dtype=torch.float32),
        )
        nb_sessions = len(act)
        lb, tau, ub, gamma = 20, 60, 100, 0.8

        alpha = torch.zeros(
            [
                nb_sessions,
                nb_chains,
                act.shape[-1],
                self.nb_blocklengths,
                self.nb_typeblocks,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        alpha[:, :, 0, 0, :] = 1.0 / 3
        alpha = alpha.reshape(
            nb_sessions, nb_chains, -1, self.nb_typeblocks * self.nb_blocklengths
        )
        h = torch.zeros(
            [nb_sessions, nb_chains, self.nb_typeblocks * self.nb_blocklengths],
            device=self.device,
            dtype=torch.float32,
        )

        if self.single_zeta:
            zeta_pos, zeta_neg = zeta, zeta

        lapses, zetas = mut.get_parameters(
            lapse_pos, lapse_neg, side, zeta_pos, zeta_neg
        )

        # build transition matrix
        b = torch.zeros(
            [self.nb_blocklengths, 3, 3], device=self.device, dtype=torch.float32
        )
        b[1:][:, 0, 0], b[1:][:, 1, 1], b[1:][:, 2, 2] = 1, 1, 1  # case when l_t > 0
        b[0][0][-1], b[0][-1][0], b[0][1][np.array([0, 2])] = (
            1,
            1,
            1.0 / 2,
        )  # case when l_t = 1
        n = torch.arange(
            1, self.nb_blocklengths + 1, device=self.device, dtype=torch.float32
        )
        ref = torch.exp(-n / tau) * (lb <= n) * (ub >= n)
        hazard = torch.cummax(
            ref / torch.flip(torch.cumsum(torch.flip(ref, (0,)), 0) + 1e-18, (0,)), 0
        )[0]
        padding = torch.zeros(
            self.nb_blocklengths - 1, device=self.device, dtype=torch.float32
        )
        l = torch.cat(
            (
                torch.unsqueeze(hazard, -1),
                torch.cat((torch.diag(1 - hazard[:-1]), padding[np.newaxis]), axis=0),
            ),
            axis=-1,
        )  # l_{t-1}, l_t
        transition = 1e-12 + torch.transpose(
            l[:, :, np.newaxis, np.newaxis] * b[np.newaxis], 1, 2
        ).reshape(self.nb_typeblocks * self.nb_blocklengths, -1)

        # likelihood
        ones = torch.ones(
            (nb_sessions, act.shape[-1]), device=self.device, dtype=torch.float32
        )
        lks = torch.stack(
            [
                gamma * (side == -1) + (1 - gamma) * (side == 1),
                ones * 1.0 / 2,
                gamma * (side == 1) + (1 - gamma) * (side == -1),
            ]
        ).T
        to_update = torch.unsqueeze(torch.unsqueeze(act != 0, -1), -1) * 1

        for i_trial in range(act.shape[-1]):
            if i_trial >= 0:  # python indexing starts at 0
                if i_trial > 0:
                    alpha[:, :, i_trial] = torch.sum(
                        torch.unsqueeze(h, -1) * transition, axis=2
                    ) * to_update[:, i_trial - 1] + alpha[:, :, i_trial - 1] * (
                        1 - to_update[:, i_trial - 1]
                    )
                # else:
                #    alpha = torch.zeros(
                #        [nb_sessions, nb_chains, act.shape[-1], self.nb_blocklengths, self.nb_typeblocks],
                #        device=self.device, dtype=torch.float32)
                #    alpha[:, :, i_trial, 0, 0] = 0.5
                #    alpha[:, :, i_trial, 0, -1] = 0.5
                #    alpha = alpha.reshape(nb_sessions, nb_chains, -1, self.nb_typeblocks * self.nb_blocklengths)
                h = alpha[:, :, i_trial] * torch.unsqueeze(lks[i_trial], 1).repeat(
                    1, 1, self.nb_blocklengths
                )
                h = h / torch.unsqueeze(torch.sum(h, axis=-1), -1)

        predictive = torch.sum(
            alpha.reshape(
                nb_sessions, nb_chains, -1, self.nb_blocklengths, self.nb_typeblocks
            ),
            3,
        )
        Pis = (
            predictive[:, :, :, 0] * gamma
            + predictive[:, :, :, 1] * 0.5
            + predictive[:, :, :, 2] * (1 - gamma)
        )

        logp_ch, pActions, prediction_error = mut.compute_logp_ch_and_pe(
            (1 - Pis),
            stim,
            act,
            side,
            zetas,
            lapses,
            (None if not self.repetition_bias else rep_bias),
        )
        priors = 1 - torch.tensor(Pis.detach(), device="cpu")

        if return_details:
            return logp_ch, priors, pActions
        return np.array(torch.sum(logp_ch, axis=(0, -1)))
