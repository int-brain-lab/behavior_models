from models import model
import torch
import numpy as np
from torch.distributions.normal import Normal
from models import utils as mut

unsqueeze = lambda x : torch.unsqueeze(torch.unsqueeze(x, 0), -1)

class expSmoothing_prevAction(model.Model):
    '''
        Model where the prior is based on an exponential estimation of the previous stimulus side
    '''
    name = 'actKernel'

    def __init__(self, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side, single_zeta):
        name = 'actKernel' + '_single_zeta' * single_zeta
        self.single_zeta = single_zeta
        nb_params = 4 + (not single_zeta) * 1
        lb_params, ub_params = np.zeros(nb_params), np.concatenate((np.ones(nb_params - 2), np.array([.5, .5])))
        std_RW = np.concatenate((np.array([0.05]), np.ones(1 + (not single_zeta) * 1) * 0.04, np.array([0.02, 0.02])))
        super().__init__(name, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side, nb_params, lb_params, ub_params, std_RW)

    def compute_lkd(self, arr_params, act, stim, side, return_details):
        '''
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
        '''
        nb_chains = len(arr_params)
        if self.single_zeta:
            alpha, zeta, lapse_pos, lapse_neg = torch.tensor(arr_params, device=self.device,
                                                                    dtype=torch.float32).T
        else:
            alpha, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(arr_params, device=self.device,
                                                                    dtype=torch.float32).T
        loglikelihood = np.zeros(nb_chains)
        act, stim, side = torch.tensor(act, dtype=torch.float32), torch.tensor(stim, dtype=torch.float32), torch.tensor(side, dtype=torch.float32)
        nb_sessions = len(act)

        values = torch.zeros([nb_sessions, nb_chains, act.shape[-1], 2], dtype=torch.float32) + 0.5

        alpha = unsqueeze(alpha)
        lapses = (
            unsqueeze(lapse_pos) * (torch.unsqueeze(side,1) > 0) +
            unsqueeze(lapse_neg) * (torch.unsqueeze(side, 1) < 0) +
            0.5 * (unsqueeze(lapse_neg) + unsqueeze(lapse_pos)) * (torch.unsqueeze(side, 1) == 0)
        )

        if self.single_zeta:
            zeta_pos, zeta_neg = zeta, zeta

        zetas = (
                unsqueeze(zeta_pos) * (torch.unsqueeze(side, 1) > 0) +
                unsqueeze(zeta_neg) * (torch.unsqueeze(side, 1) < 0) +
                0.5 * (unsqueeze(zeta_pos) + unsqueeze(zeta_neg)) * (torch.unsqueeze(side, 1) == 0)
        )

        for t in range(act.shape[-1]):
            if t > 0:
                a_prev = torch.stack([act[:, t - 1]==-1, act[:, t - 1] == 1]) * 1
                values[act[:, t-1] != 0, :, t] = ((1 - alpha) * values[act[:, t - 1] != 0, :, t - 1] +
                                                  alpha * torch.unsqueeze(a_prev.T[act[:, t - 1] != 0], 1))
                values[act[:, t-1] == 0, :, t] = values[act[:, t-1] == 0, :, t - 1]

        # assert(torch.max(torch.abs(torch.sum(values, axis=-1) - 1)) < 1e-5)

        values = torch.clamp(values, min=1e-6, max=1 - 1e-6)
        pLeft = mut.combine_lkd_prior(stim, zetas, values[:, :, :, 1], lapses)
        pRight = 1 - pLeft # mut.combine_lkd_prior(-stim, zetas, values[:, :, :, 0], lapses)
        #assert (torch.max(torch.abs(pLeft + pRight - 1)) < 1e-5)
        pActions = torch.stack((pRight/(pRight + pLeft), pLeft/(pRight + pLeft)))

        belief = pActions[0] * (torch.unsqueeze(act, 1) == -1) + pActions[1] * (torch.unsqueeze(act, 1) == 1)
        correct = (act == side) * 1
        prediction_error = (torch.unsqueeze(correct, 1) - belief)

        pActions = pActions * (1 - torch.unsqueeze(lapses, 0)) + torch.unsqueeze(lapses, 0) / 2.

        p_ch = pActions[0] * (torch.unsqueeze(act, 1) == -1) + pActions[1] * (torch.unsqueeze(act, 1) == 1) + 1 * (torch.unsqueeze(act, 1) == 0) # discard trials where agent did not answer
        p_ch = torch.minimum(torch.maximum(p_ch, torch.tensor(1e-8)), torch.tensor(1 - 1e-8))
        logp_ch = torch.log(p_ch)
        if return_details:
            return logp_ch, values[:, :, :, 1], prediction_error
        return np.array(torch.sum(logp_ch, axis=(0, -1)))

    def simulate(self, arr_params, stim, side, valid, nb_simul=50, only_perf=True, return_prior=False):
        '''
        custom
        '''
        assert(stim.shape == side.shape), 'side and stim don\'t have the same shape'

        nb_chains = len(arr_params)
        if arr_params.shape[-1] == 4:
            alpha, zeta, lapse_pos, lapse_neg = torch.tensor(arr_params).T
        else:
            raise NotImplementedError
        stim, side = torch.tensor(stim), torch.tensor(side)
        nb_sessions = len(stim)

        act_sim = torch.zeros(nb_sessions, nb_chains, stim.shape[-1], nb_simul)
        values = torch.zeros([nb_sessions, nb_chains, stim.shape[-1], nb_simul, 2], dtype=torch.float64) + 0.5

        valid_arr = np.tile(valid[:, np.newaxis,:,np.newaxis], (1, nb_chains, 1, nb_simul))

        alpha = torch.unsqueeze(unsqueeze(alpha), -1)
        zetas = unsqueeze(zeta) * (torch.unsqueeze(side,1) > 0) + unsqueeze(zeta) * (torch.unsqueeze(side,1) <= 0)
        lapses = (
            unsqueeze(lapse_pos) * (torch.unsqueeze(side,1) > 0) +
            unsqueeze(lapse_neg) * (torch.unsqueeze(side, 1) < 0) +
            0.5 * (unsqueeze(lapse_neg) + unsqueeze(lapse_pos)) * (torch.unsqueeze(side, 1) == 0)
        )
        Rho = torch.unsqueeze(torch.minimum(torch.maximum(Normal(loc=torch.unsqueeze(stim, 1), scale=zetas).cdf(torch.tensor(0)),
                                                          torch.tensor(1e-7)), torch.tensor(1 - 1e-7)), -1) # pRight likelihood

        for t in range(stim.shape[-1]):
            if t > 0:
                a_prev = torch.stack([act_sim[:, :, t - 1].T==-1, act_sim[:, :, t - 1].T==1]) * 1
                values[:, :, t] = (1 - alpha) * values[:, :, t-1] + alpha * a_prev.T
            pRight, pLeft = values[:, :, t, :, 0] * Rho[:, :, t], values[:, :, t, :, 1] * (1 - Rho[:, :, t])
            pActions = torch.stack((pRight/(pRight + pLeft), pLeft/(pRight + pLeft)))
            pActions = pActions * (1 - lapses[:, :, t]) + lapses[:, :, t] / 2.
            act_sim[:, :, t] = 2 * (torch.rand(nb_sessions, nb_chains, nb_simul) < pActions[1]) - 1

        assert(torch.max(torch.abs(torch.sum(values, axis=-1) - 1)) < 1e-6)

        correct = (act_sim == side[:, np.newaxis, :, np.newaxis])
        correct = np.array(correct, dtype=np.float)
        correct[valid_arr == False] = np.nan
        perf = np.nanmean(correct, axis=(0, -2, -1))

        if only_perf:
            return perf
        elif return_prior:
            return act_sim, stim, side, values[:, :, :, :, 1]
        else:
            return act_sim, stim, side

