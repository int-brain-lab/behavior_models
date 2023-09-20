import numpy as np
import torch
from torch.distributions import Normal

from iblutil.util import setup_logger

from behavior_models import utils as mut, base_models

logger = setup_logger("ibl")


class OptimalBayesian(base_models.PriorModel):
    """
    Model where the prior is based on an exponential estimation of the previous stimulus side
    """

    name = "optBay"

    def __init__(self, *args, single_zeta=True, repetition_bias=False, with_unbiased=False, single_lapserate=False, **kwargs):
        self.name = (
                "optimal_bayesian"
                + "_with_repBias" * repetition_bias
                + "_single_zeta" * single_zeta
                + "_with_unbiased" * with_unbiased
                + "_single_lapserate" * single_lapserate
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

        self.with_unbiased = with_unbiased
        self.single_lapserate = single_lapserate

        super(OptimalBayesian, self).__init__(*args, nb_params=nb_params, lb_params=lb_params, ub_params=ub_params, std_RW=std_RW, **kwargs)
        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda:0")
            logger.info("Using GPU")

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
        alpha[:, :, 0, 0, 1] = 1.0
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

        if self.single_lapserate:
            lapses, zetas = mut.get_parameters(
                lapse_pos, lapse_pos, side, zeta_pos, zeta_neg
            )
        else:
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
            if (i_trial > 0 and not self.with_unbiased) or (self.with_unbiased and i_trial > 90):
                alpha[:, :, i_trial] = torch.sum(
                    torch.unsqueeze(h, -1) * transition, axis=2
                ) * to_update[:, i_trial - 1] + alpha[:, :, i_trial - 1] * (
                                               1 - to_update[:, i_trial - 1]
                                       )
            elif (self.with_unbiased and i_trial == 90):
                alpha = torch.zeros(
                    [nb_sessions, nb_chains, act.shape[-1], self.nb_blocklengths, self.nb_typeblocks],
                    device=self.device, dtype=torch.float32)
                alpha[:, :, :i_trial, 0, 1] = 1.0
                alpha[:, :, i_trial, 0, 0] = 0.5
                alpha[:, :, i_trial, 0, -1] = 0.5
                alpha = alpha.reshape(nb_sessions, nb_chains, -1, self.nb_typeblocks * self.nb_blocklengths)
            elif self.with_unbiased:
                alpha[:, :, i_trial] = alpha[:, :, i_trial - 1]
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
        return torch.sum(logp_ch, axis=(0, -1)).cpu().numpy()


class ActionKernel(base_models.PriorModel):
    """
    Model where the prior is based on an exponential estimation of the previous stimulus side
    """

    name = "actKernel"

    def __init__(self, *args, single_zeta=True, repetition_bias=None, **kwargs):
        self.name = "actKernel" + "_single_zeta" * single_zeta
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
        self.repetition_bias = None  # this model can't have a repetition bias
        super(ActionKernel, self).__init__(*args, nb_params=nb_params, lb_params=lb_params, ub_params=ub_params,  std_RW=std_RW, **kwargs)


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
                values[act[:, t - 1] != 0, :, t] = (1 - alpha) * values[
                    act[:, t - 1] != 0, :, t - 1
                ] + alpha * torch.unsqueeze(a_prev.T[act[:, t - 1] != 0], 1)
                values[act[:, t - 1] == 0, :, t] = values[act[:, t - 1] == 0, :, t - 1]

        logp_ch, pActions, prediction_error = mut.compute_logp_ch_and_pe(
            values[:, :, :, 1], stim, act, side, zetas, lapses, repetition_bias=None
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
        correct = np.array(correct, dtype=float)
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


class StimulusKernel(base_models.PriorModel):
    """
    Model where the prior is based on an exponential estimation of the previous stimulus side
    """

    name = "stimKernel"

    def __init__(self, *args, single_zeta=True, repetition_bias=False, **kwargs):
        self.name = (
            "stimKernel"
            + "_with_repBias" * repetition_bias
            + "_single_zeta" * single_zeta
        )
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
        self.repetition_bias = repetition_bias
        if repetition_bias:
            lb_params, ub_params = np.append(lb_params, 0), np.append(ub_params, 0.5)
            std_RW = np.append(std_RW, 0.01)
            nb_params += 1

        super(StimulusKernel, self).__init__(
            *args, nb_params=nb_params, lb_params=lb_params, ub_params=ub_params, std_RW=std_RW, **kwargs)

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
        if not self.repetition_bias and self.single_zeta:
            alpha, zeta, lapse_pos, lapse_neg = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        if self.repetition_bias and self.single_zeta:
            alpha, zeta, lapse_pos, lapse_neg, rep_bias = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        if not self.repetition_bias and not self.single_zeta:
            alpha, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        if self.repetition_bias and not self.single_zeta:
            alpha, zeta_pos, zeta_neg, lapse_pos, lapse_neg, rep_bias = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        act, stim, side = (
            torch.tensor(act, dtype=torch.float32),
            torch.tensor(stim, dtype=torch.float32),
            torch.tensor(side, dtype=torch.float32),
        )
        nb_sessions = len(act)

        values = (
            torch.zeros([nb_sessions, nb_chains, act.shape[-1], 2], dtype=torch.float64)
            + 0.5
        )

        alpha = mut.unsqueeze(alpha)

        if self.single_zeta:
            zeta_pos, zeta_neg = zeta, zeta

        lapses, zetas = mut.get_parameters(
            lapse_pos, lapse_neg, side, zeta_pos, zeta_neg
        )

        for t in range(act.shape[-1]):
            if t > 0:
                s_prev = torch.stack([side[:, t - 1] == -1, side[:, t - 1] == 1]) * 1
                values[act[:, t - 1] != 0, :, t] = (1 - alpha) * values[
                    act[:, t - 1] != 0, :, t - 1
                ] + alpha * torch.unsqueeze(s_prev.T[act[:, t - 1] != 0], 1)
                values[act[:, t - 1] == 0, :, t] = values[act[:, t - 1] == 0, :, t - 1]

        rep_bias = None if not self.repetition_bias else rep_bias
        logp_ch, pActions, prediction_error = mut.compute_logp_ch_and_pe(
            values[:, :, :, 1], stim, act, side, zetas, lapses, rep_bias
        )

        if return_details:
            return logp_ch, values[:, :, :, 1], pActions
        return np.array(torch.sum(logp_ch, axis=(0, -1)))


class StimulusKernel_4alphas(base_models.PriorModel):
    """
    Model where the prior is based on an exponential estimation of the previous stimulus side
    """
    def __init__(self, *args, single_zeta=True, repetition_bias=False, **kwargs):
        self.name = (
            "expSmoothingStimSides_4alphas"
            + "_with_repBias" * repetition_bias
            + "_single_zeta" * single_zeta
        )
        if single_zeta:
            nb_params, lb_params, ub_params = (
                7,
                np.array([0, 0, 0, 0, 0, 0, 0]),
                np.array([1, 1, 1, 1, 1, 0.5, 0.5]),
            )
            std_RW = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
        else:
            nb_params, lb_params, ub_params = (
                8,
                np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([1, 1, 1, 1, 1, 1, 0.5, 0.5]),
            )
            std_RW = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
        self.repetition_bias = repetition_bias
        self.single_zeta = single_zeta
        if repetition_bias:
            nb_params += 1
            lb_params, ub_params = np.append(lb_params, 0), np.append(ub_params, 0.5)
            std_RW = np.append(std_RW, 0.01)
        super(StimulusKernel_4alphas, self).__init__(
            *args, nb_params=nb_params, lb_params=lb_params, ub_params=ub_params, std_RW=std_RW, **kwargs)

    def compute_lkd(self, arr_params, act, stim, side, return_details):
        nb_chains = len(arr_params)
        if not self.repetition_bias and not self.single_zeta:
            (
                alpha_ch_rew,
                alpha_ch_unrew,
                alpha_unch_rew,
                alpha_unch_unrew,
                zeta_pos,
                zeta_neg,
                lapse_pos,
                lapse_neg,
            ) = torch.tensor(arr_params).T
        if not self.single_zeta and self.repetition_bias:
            (
                alpha_ch_rew,
                alpha_ch_unrew,
                alpha_unch_rew,
                alpha_unch_unrew,
                zeta_pos,
                zeta_neg,
                lapse_pos,
                lapse_neg,
                rep_bias,
            ) = torch.tensor(arr_params).T
        if self.single_zeta and not self.repetition_bias:
            (
                alpha_ch_rew,
                alpha_ch_unrew,
                alpha_unch_rew,
                alpha_unch_unrew,
                zeta,
                lapse_pos,
                lapse_neg,
            ) = torch.tensor(arr_params).T
        if self.single_zeta and self.repetition_bias:
            (
                alpha_ch_rew,
                alpha_ch_unrew,
                alpha_unch_rew,
                alpha_unch_unrew,
                zeta,
                lapse_pos,
                lapse_neg,
                rep_bias,
            ) = torch.tensor(arr_params).T
        act, stim, side = torch.tensor(act), torch.tensor(stim), torch.tensor(side)
        nb_sessions = len(act)

        values = (
            torch.zeros([nb_sessions, nb_chains, act.shape[-1], 2], dtype=torch.float64)
            + 0.5
        )

        if self.single_zeta:
            zeta_pos, zeta_neg = zeta, zeta

        lapses, zetas = mut.get_parameters(
            lapse_pos, lapse_neg, side, zeta_pos, zeta_neg
        )
        rew = torch.unsqueeze(torch.unsqueeze((act == side) * 1, -1), -1)

        for t in range(act.shape[-1]):
            if t > 0:

                s_prev = torch.stack([side[:, t - 1] == -1, side[:, t - 1] == 1]) * 1
                a_prev = torch.stack([act[:, t - 1] == -1, act[:, t - 1] == 1]).T * 1

                alphas = (
                    torch.unsqueeze(a_prev, 1)
                    * mut.unsqueeze(alpha_ch_rew)
                    * rew[:, t - 1]
                    + torch.unsqueeze(1 - a_prev, 1)
                    * mut.unsqueeze(alpha_unch_rew)
                    * rew[:, t - 1]
                    + torch.unsqueeze(a_prev, 1)
                    * mut.unsqueeze(alpha_ch_unrew)
                    * (1 - rew[:, t - 1])
                    + torch.unsqueeze(1 - a_prev, 1)
                    * mut.unsqueeze(alpha_unch_unrew)
                    * (1 - rew[:, t - 1])
                )

                values[:, :, t] = (1 - alphas) * values[
                    :, :, t - 1
                ] + alphas * torch.unsqueeze(s_prev.T, 1)
                values[:, :, t] = values[:, :, t] / torch.unsqueeze(
                    torch.sum(values[:, :, t], axis=-1), -1
                )
                values[act[:, t - 1] == 0, :, t] = values[act[:, t - 1] == 0, :, t - 1]

        logp_ch, pActions, prediction_error = mut.compute_logp_ch_and_pe(
            values[:, :, :, 1], stim, act, side, zetas, lapses, repetition_bias=None
        )

        if return_details:
            return logp_ch, values[:, :, :, 1], pActions
        return np.array(torch.sum(logp_ch, axis=(0, -1)))


