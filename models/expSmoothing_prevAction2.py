from models import model
import torch
import numpy as np
from torch.distributions.normal import Normal
from models import utils as mut


class expSmoothing_prevAction2(model.Model):
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
    ):
        name = "expSmoothing_prevAction2_lr_constrained" + "_single_zeta" * single_zeta
        if single_zeta:
            nb_params, lb_params, ub_params = (
                6,
                np.array([0, 0, 0, 0, 0, 0]),
                np.array([1, 1, 1, 1, 0.5, 0.5]),
            )
            std_RW = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
        else:
            nb_params, lb_params, ub_params = (
                7,
                np.array([0, 0, 0, 0, 0, 0, 0]),
                np.array([1, 1, 1, 1, 1, 0.5, 0.5]),
            )
            std_RW = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
        self.single_zeta = single_zeta
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
        nb_chains = len(arr_params)
        if self.single_zeta:
            (
                weights,
                alpha_a,
                alpha_s,
                zeta,
                lapse_pos,
                lapse_neg,
            ) = torch.tensor(arr_params).T
        else:
            (
                weights,
                alpha_a,
                alpha_s,
                zeta_pos,
                zeta_neg,
                lapse_pos,
                lapse_neg,
            ) = torch.tensor(arr_params).T
        act, stim, side = torch.tensor(act), torch.tensor(stim), torch.tensor(side)
        nb_sessions = len(act)

        values_a = (
            torch.zeros([nb_sessions, nb_chains, act.shape[-1], 2], dtype=torch.float64)
            + 0.5
        )
        values_s = (
            torch.zeros([nb_sessions, nb_chains, act.shape[-1], 2], dtype=torch.float64)
            + 0.5
        )

        if self.single_zeta:
            zeta_pos, zeta_neg = zeta, zeta

        alpha_a, alpha_s = mut.unsqueeze(alpha_a), mut.unsqueeze(alpha_s)
        alpha_s = alpha_a + alpha_s * (1 - alpha_a)  # make alpha_s greater than alpha_a
        lapses, zetas = mut.get_parameters(
            lapse_pos, lapse_neg, side, zeta_pos, zeta_neg
        )

        for t in range(act.shape[-1]):
            if t > 0:
                # previous actions
                a_prev = torch.stack([act[:, t - 1] == -1, act[:, t - 1] == 1]) * 1
                values_a[:, :, t] = (1 - alpha_a) * values_a[
                    :, :, t - 1
                ] + alpha_a * torch.unsqueeze(a_prev.T, 1)
                values_a[act[:, t - 1] == 0, :, t] = values_a[
                    act[:, t - 1] == 0, :, t - 1
                ]

                # previous actions 2
                a_prev2 = torch.stack([act[:, t - 1] == -1, act[:, t - 1] == 1]) * 1
                values_s[:, :, t] = (1 - alpha_s) * values_s[
                    :, :, t - 1
                ] + alpha_s * torch.unsqueeze(a_prev2.T, 1)
                values_s[act[:, t - 1] == 0, :, t] = values_s[
                    act[:, t - 1] == 0, :, t - 1
                ]

        assert torch.max(torch.abs(torch.sum(values_s, axis=-1) - 1)) < 1e-6
        assert torch.max(torch.abs(torch.sum(values_a, axis=-1) - 1)) < 1e-6

        values = values_s * torch.unsqueeze(mut.unsqueeze(weights), -1) + values_a * (
            1 - torch.unsqueeze(mut.unsqueeze(weights), -1)
        )

        logp_ch, pActions, prediction_error = mut.compute_logp_ch_and_pe(
            values[:, :, :, 1], stim, act, side, zetas, lapses, repetition_bias=None
        )

        if return_details:
            return logp_ch, values[:, :, :, 1], pActions
        return np.array(torch.sum(logp_ch, axis=(0, -1)))

    def simulate(self, arr_params, stim, side, valid, nb_simul=50):
        """
        custom
        """
        return NotImplementedError

        assert stim.shape == side.shape, "side and stim don't have the same shape"

        nb_chains = len(arr_params)
        if arr_params.shape[-1] == 5:
            alpha, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(arr_params).T
        else:
            raise NotImplementedError
        stim, side = torch.tensor(stim), torch.tensor(side)
        nb_sessions = len(stim)

        act_sim = torch.zeros(nb_sessions, nb_chains, stim.shape[-1], nb_simul)
        values = (
            torch.zeros(
                [nb_sessions, nb_chains, stim.shape[-1], nb_simul, 2],
                dtype=torch.float64,
            )
            + 0.5
        )

        valid_arr = np.tile(
            valid[:, np.newaxis, :, np.newaxis], (1, nb_chains, 1, nb_simul)
        )

        alpha = torch.unsqueeze(unsqueeze(alpha), -1)
        zetas = unsqueeze(zeta_pos) * (torch.unsqueeze(side, 1) > 0) + unsqueeze(
            zeta_neg
        ) * (torch.unsqueeze(side, 1) <= 0)
        lapses = torch.unsqueeze(
            unsqueeze(lapse_pos) * (torch.unsqueeze(side, 1) > 0)
            + unsqueeze(lapse_neg) * (torch.unsqueeze(side, 1) <= 0),
            -1,
        )
        Rho = torch.unsqueeze(
            torch.minimum(
                torch.maximum(
                    Normal(loc=torch.unsqueeze(stim, 1), scale=zetas).cdf(0),
                    torch.tensor(1e-7),
                ),
                torch.tensor(1 - 1e-7),
            ),
            -1,
        )  # pRight likelihood

        for t in range(stim.shape[-1]):
            if t > 0:
                a_prev = (
                    torch.stack(
                        [act_sim[:, :, t - 1].T == -1, act_sim[:, :, t - 1].T == 1]
                    )
                    * 1
                )
                values[:, :, t] = (1 - alpha) * values[:, :, t - 1] + alpha * a_prev.T
            pRight, pLeft = values[:, :, t, :, 0] * Rho[:, :, t], values[
                :, :, t, :, 1
            ] * (1 - Rho[:, :, t])
            pActions = torch.stack(
                (pRight / (pRight + pLeft), pLeft / (pRight + pLeft))
            )
            pActions = pActions * (1 - lapses[:, :, t]) + lapses[:, :, t] / 2.0
            act_sim[:, :, t] = (
                2 * (torch.rand(nb_sessions, nb_chains, nb_simul) < pActions[1]) - 1
            )

        assert torch.max(torch.abs(torch.sum(values, axis=-1) - 1)) < 1e-6

        correct = act_sim == side[:, np.newaxis, :, np.newaxis]
        correct = np.array(correct, dtype=np.float)
        correct[valid_arr == False] = np.nan
        perf = np.nanmean(correct, axis=(0, -2, -1))

        return perf
