from behavior_models.models import model
import torch
import numpy as np
from torch.distributions.normal import Normal
from behavior_models.models import utils as mut


class expSmoothing_prevAction_beginningSession(model.Model):
    """
    Model where the prior is based on an exponential estimation of the previous stimulus side
    """

    name = "actKernel_beginningSession"

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
        with_choice_trace=False,
    ):
        if (not single_zeta) or repetition_bias or with_choice_trace:
            raise NotImplementedError('This case is not implemented')
        name = "actKernel_beginningSession" + "_single_zeta" * single_zeta + "_with_repBias" * repetition_bias + "_with_choiceTrace" * with_choice_trace
        if with_choice_trace:
            if (not repetition_bias) or (not single_zeta):
                raise AssertionError('repBias must be True if choice trace is True')

        self.single_zeta = single_zeta
        nb_params = 6 + (not single_zeta) * 1
        lb_params, ub_params = np.zeros(nb_params), np.concatenate(
            (np.ones(nb_params - 2), np.array([0.5, 0.5]))
        )
        std_RW = np.concatenate(
            (
                np.array([0.05, 0.05, 0.05]),
                np.ones(1 + (not single_zeta) * 1) * 0.04,
                np.array([0.02, 0.02]),
            )
        )

        self.repetition_bias = repetition_bias
        if repetition_bias:
            lb_params, ub_params = np.append(lb_params, 0), np.append(ub_params, 0.5)
            std_RW = np.append(std_RW, 0.01)
            nb_params += 1
        self.with_choice_trace = with_choice_trace
        if with_choice_trace:
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
        alpha_0_90, alpha_90_180, alpha_180_10000, zeta, lapse_pos, lapse_neg = torch.tensor(
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

        alpha_0_90 = mut.unsqueeze(alpha_0_90)
        alpha_90_180 = mut.unsqueeze(alpha_90_180)
        alpha_180_10000 = mut.unsqueeze(alpha_180_10000)
        lapses, zetas = mut.get_parameters(
            lapse_pos, lapse_neg, side, zeta_pos, zeta_neg
        )

        for t in range(act.shape[-1]):
            if t > 0:
                if t <= 90:
                    alpha = alpha_0_90
                elif t <= 180:
                    alpha = alpha_90_180
                else:
                    alpha = alpha_180_10000
                a_prev = torch.stack([act[:, t - 1] == -1, act[:, t - 1] == 1]) * 1
                values[act[:, t - 1] != 0, :, t] = (1 - alpha) * values[
                    act[:, t - 1] != 0, :, t - 1
                ] + alpha * torch.unsqueeze(a_prev.T[act[:, t - 1] != 0], 1)
                values[act[:, t - 1] == 0, :, t] = values[act[:, t - 1] == 0, :, t - 1]

        logp_ch, pActions, prediction_error = mut.compute_logp_ch_and_pe(
            values[:, :, :, 1], stim, act, side, zetas, lapses, repetition_bias=None,
            choice_trace=None,
        )

        if return_details:
            return logp_ch, values[:, :, :, 1], pActions
        return np.array(torch.sum(logp_ch, axis=(0, -1)))

    def simulate(
        self,
        arr_params,
        stim,
        side,
        nb_simul=50,
        only_perf=True,
        return_prior=False,
        actions=None,
    ):
        """
        custom
        """
        assert stim.shape == side.shape, "side and stim don't have the same shape"

        if self.single_zeta:
            alpha, zeta, lapse_pos, lapse_neg = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        else:
            alpha, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        stim, side = torch.tensor(stim), torch.tensor(side)

        act_sim = torch.zeros(stim.shape[-1], nb_simul)
        values = torch.zeros([stim.shape[-1], nb_simul, 2], dtype=torch.float64) + 0.5

        lapses = (
            lapse_pos * (side > 0)
            + lapse_neg * (side < 0)
            + 0.5 * (lapse_neg + lapse_pos) * (side == 0)
        )

        if self.single_zeta:
            zeta_pos, zeta_neg = zeta, zeta

        zetas = (
            zeta_pos * (side > 0)
            + zeta_neg * (side < 0)
            + 0.5 * (zeta_pos + zeta_neg) * (side == 0)
        )

        for t in range(stim.shape[-1]):
            if t > 0:
                if actions is None:
                    a_prev = (
                        torch.stack([act_sim[t - 1] == -1, act_sim[t - 1] == 1]) * 1
                    )
                    values[t] = (1 - alpha) * values[t - 1] + alpha * a_prev.T
                else:
                    if actions[t - 1] != 0:
                        a_prev = (
                            torch.stack(
                                [
                                    torch.from_numpy(actions)[t - 1] == -1,
                                    torch.from_numpy(actions)[t - 1] == 1,
                                ]
                            )
                            * 1
                        )
                        values[t] = (1 - alpha) * values[t - 1] + alpha * a_prev.T
                    else:
                        values[t] = values[t - 1]
            values = torch.clamp(values, min=1e-7, max=1 - 1e-7)
            pLeft = mut.combine_lkd_prior(
                torch.tensor([stim[t]]),
                zetas[t],
                values[t, :, 1],
            )
            pLeft = pLeft * (1 - lapses[t]) + lapses[t] / 2.0
            pActions = torch.vstack((1 - pLeft, pLeft))
            act_sim[t] = 2 * (torch.rand(nb_simul) < pActions[1]) - 1

        correct = act_sim == side[:, None]
        correct = np.array(correct, dtype=np.float)
        perf = np.nanmean(correct, axis=(0,))

        if only_perf:
            return perf
        elif return_prior:
            return act_sim, stim, side, values[:, :, 1]
        else:
            return act_sim, stim, side

    def simulate_parallel(
        self,
        arr_params,
        stim,
        side,
        nb_simul=50,
    ):
        """
        custom
        """
        assert stim.shape == side.shape, "side and stim don't have the same shape"

        if self.single_zeta:
            alpha, zeta, lapse_pos, lapse_neg = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        else:
            alpha, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        stim, side = torch.tensor(stim), torch.tensor(side)

        nb_sessions = len(stim)
        act_sim = torch.zeros(nb_sessions, stim.shape[-1], nb_simul)
        values = (
            torch.zeros([nb_sessions, stim.shape[-1], nb_simul, 2], dtype=torch.float64)
            + 0.5
        )

        lapses = (
            lapse_pos * (side > 0)
            + lapse_neg * (side < 0)
            + 0.5 * (lapse_neg + lapse_pos) * (side == 0)
        )

        if self.single_zeta:
            zeta_pos, zeta_neg = zeta, zeta

        zetas = (
            zeta_pos * (side > 0)
            + zeta_neg * (side < 0)
            + 0.5 * (zeta_pos + zeta_neg) * (side == 0)
        )

        for t in range(stim.shape[-1]):
            if t > 0:
                a_prev = torch.swapaxes(
                    torch.stack([act_sim[:, t - 1] == -1, act_sim[:, t - 1] == 1]) * 1,
                    1,
                    2,
                )
                values[:, t] = (1 - alpha) * values[:, t - 1] + alpha * a_prev.T
            values = torch.clamp(values, min=1e-7, max=1 - 1e-7)
            sigma_star = torch.sqrt(1 / (0.49**-2 + zetas[:, t] ** -2))
            prior_contrib = Normal(loc=0, scale=1).icdf(1 - values[:, t, :, 1])
            combined = (stim[:, t] / zetas[:, t])[:, None] - zetas[:, t][
                :, None
            ] / sigma_star[:, None] * prior_contrib
            pLeft = Normal(loc=0, scale=1).cdf(combined)
            pLeft = pLeft * (1 - lapses[:, t, None]) + lapses[:, t, None] / 2.0
            act_sim[:, t] = 2 * (torch.rand_like(pLeft) < pLeft) - 1

        return act_sim, stim, side
