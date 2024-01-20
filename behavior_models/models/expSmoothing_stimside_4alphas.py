from behavior_models.models import model
import torch
import numpy as np
from behavior_models.models import utils as mut


class expSmoothing_stimside_4alphas(model.Model):
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
        with_choice_trace=False,
    ):
        name = (
            "expSmoothingStimSides_4alphas"
            + "_with_repBias" * repetition_bias
            + "_single_zeta" * single_zeta
            + "_with_choiceTrace" * with_choice_trace
        )
        if with_choice_trace:
            if (not repetition_bias) or (not single_zeta):
                raise AssertionError('repBias must be True if choice trace is True')

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
        nb_chains = len(arr_params)
        if self.with_choice_trace:
            (
                alpha_ch_rew,
                alpha_ch_unrew,
                alpha_unch_rew,
                alpha_unch_unrew,
                zeta,
                lapse_pos,
                lapse_neg,
                rep_bias,
                choice_trace_lr,
            ) = torch.tensor(arr_params, dtype=torch.float32).T
        elif not self.repetition_bias and not self.single_zeta:
            (
                alpha_ch_rew,
                alpha_ch_unrew,
                alpha_unch_rew,
                alpha_unch_unrew,
                zeta_pos,
                zeta_neg,
                lapse_pos,
                lapse_neg,
            ) = torch.tensor(arr_params, dtype=torch.float32).T
        elif not self.single_zeta and self.repetition_bias:
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
            ) = torch.tensor(arr_params, dtype=torch.float32).T
        elif self.single_zeta and not self.repetition_bias:
            (
                alpha_ch_rew,
                alpha_ch_unrew,
                alpha_unch_rew,
                alpha_unch_unrew,
                zeta,
                lapse_pos,
                lapse_neg,
            ) = torch.tensor(arr_params, dtype=torch.float32).T
        elif self.single_zeta and self.repetition_bias:
            (
                alpha_ch_rew,
                alpha_ch_unrew,
                alpha_unch_rew,
                alpha_unch_unrew,
                zeta,
                lapse_pos,
                lapse_neg,
                rep_bias,
            ) = torch.tensor(arr_params, dtype=torch.float32).T
        act, stim, side = torch.tensor(act), torch.tensor(stim), torch.tensor(side)
        nb_sessions = len(act)

        values = (
            torch.zeros([nb_sessions, nb_chains, act.shape[-1], 2], dtype=torch.float64)
            + 0.5
        )

        if self.with_choice_trace:
            choice_trace = torch.zeros([nb_sessions, nb_chains, act.shape[-1], 2]) + 0.5

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

                # choice trace update
                if self.with_choice_trace:
                    a_prev = torch.stack([act[:, t - 1] == -1, act[:, t - 1] == 1]) * 1
                    choice_trace[act[:, t - 1] != 0, :, t] = (1 - mut.unsqueeze(choice_trace_lr)) * choice_trace[
                        act[:, t - 1] != 0, :, t - 1
                    ] + mut.unsqueeze(choice_trace_lr) * torch.unsqueeze(a_prev.T[act[:, t - 1] != 0], 1)
                    choice_trace[act[:, t - 1] == 0, :, t] = choice_trace[act[:, t - 1] == 0, :, t - 1]

        rep_bias = None if not self.repetition_bias else rep_bias
        logp_ch, pActions, prediction_error = mut.compute_logp_ch_and_pe(
            values[:, :, :, 1], stim, act, side, zetas, lapses, rep_bias,
            choice_trace=None if not self.with_choice_trace else choice_trace,
        )

        if return_details:
            return logp_ch, values[:, :, :, 1], pActions
        return np.array(torch.sum(logp_ch, axis=(0, -1)))
