from behavior_models.models import model, utils
import torch
import numpy as np
from behavior_models.models import utils as mut
from iblutil.util import setup_logger

logger = setup_logger("ibl")

unsqueeze = lambda x: torch.unsqueeze(torch.unsqueeze(x, 0), -1)

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
        with_unbiased=False,
        single_lapserate=False,
        with_choice_trace=False,
    ):
        name = (
            "optimal_bayesian"
            + "_with_repBias" * repetition_bias
            + "_single_zeta" * single_zeta
            + "_with_unbiased" * with_unbiased
            + "_single_lapserate" * single_lapserate
            + "_with_choiceTrace" * with_choice_trace
        )
        if with_choice_trace:
            if (not repetition_bias) or with_unbiased or single_lapserate or (not single_zeta):
                raise AssertionError('repBias must be True if choice trace is True')

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

        self.with_choice_trace = with_choice_trace
        if with_choice_trace:
            lb_params, ub_params = np.append(lb_params, 0), np.append(ub_params, 0.5)
            std_RW = np.append(std_RW, 0.01)
            nb_params += 1

        self.with_unbiased = with_unbiased
        self.single_lapserate = single_lapserate

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
        if self.with_choice_trace:
            zeta, lapse_pos, lapse_neg, rep_bias, choice_trace_lr = torch.tensor(
                arr_params, device=self.device, dtype=torch.float32
            ).T
        elif not self.repetition_bias and self.single_zeta:
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

        if self.with_choice_trace:
            choice_trace = torch.zeros([nb_sessions, nb_chains, act.shape[-1], 2]) + 0.5

        for i_trial in range(act.shape[-1]):
            if (i_trial > 0 and not self.with_unbiased) or (
                self.with_unbiased and i_trial > 90
            ):
                alpha[:, :, i_trial] = torch.sum(
                    torch.unsqueeze(h, -1) * transition, axis=2
                ) * to_update[:, i_trial - 1] + alpha[:, :, i_trial - 1] * (
                    1 - to_update[:, i_trial - 1]
                )
                if self.with_choice_trace:
                    a_prev = torch.stack([act[:, i_trial - 1] == -1, act[:, i_trial - 1] == 1]) * 1
                    choice_trace[act[:, i_trial - 1] != 0, :, i_trial] = (1 - mut.unsqueeze(choice_trace_lr)) * choice_trace[
                        act[:, i_trial - 1] != 0, :, i_trial - 1
                    ] + mut.unsqueeze(choice_trace_lr) * torch.unsqueeze(a_prev.T[act[:, i_trial - 1] != 0], 1)
                    choice_trace[act[:, i_trial - 1] == 0, :, i_trial] = choice_trace[act[:, i_trial - 1] == 0, :, i_trial - 1]

            elif self.with_unbiased and i_trial == 90:
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
                alpha[:, :, :i_trial, 0, 1] = 1.0
                alpha[:, :, i_trial, 0, 0] = 0.5
                alpha[:, :, i_trial, 0, -1] = 0.5
                alpha = alpha.reshape(
                    nb_sessions,
                    nb_chains,
                    -1,
                    self.nb_typeblocks * self.nb_blocklengths,
                )
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
            choice_trace=None if not self.with_choice_trace else choice_trace,
        )
        priors = 1 - torch.tensor(Pis.detach(), device="cpu")

        if return_details:
            return logp_ch, priors, pActions
        return torch.sum(logp_ch, axis=(0, -1)).cpu().numpy()
