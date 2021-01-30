from models import model
import torch, utils
import numpy as np
from torch.distributions.normal import Normal

unsqueeze = lambda x : torch.unsqueeze(torch.unsqueeze(x, 0), -1)

class expSmoothing_prevAction(model.Model):
    '''
        Model where the prior is based on an exponential estimation of the previous stimulus side
    '''

    def __init__(self, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side):
        name = 'expSmoothingPrevActions'
        nb_params, lb_params, ub_params = 5, np.array([0, 0, 0, 0, 0]), np.array([1, 1, 1, .5, .5])
        super().__init__(name, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side, nb_params, lb_params, ub_params)

    def compute_lkd(arr_params, act, stim, side, return_details):
        nb_chains = len(arr_params)
        alpha, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(arr_params).T      
        loglikelihood = np.zeros(nb_chains)
        act, stim, side = torch.tensor(act), torch.tensor(stim), torch.tensor(side)
        nb_sessions = len(act)

        values = torch.zeros([nb_sessions, nb_chains, self.nb_trials, 2], dtype=torch.float64) + 0.5

        alpha = unsqueeze(alpha)
        zetas = unsqueeze(zeta_pos) * (torch.unsqueeze(side,1) > 0) + unsqueeze(zeta_neg) * (torch.unsqueeze(side,1) <= 0)
        lapses = unsqueeze(lapse_pos) * (torch.unsqueeze(side,1) > 0) + unsqueeze(lapse_neg) * (torch.unsqueeze(side,1) <= 0)

        for t in range(self.nb_trials):
            s = side[:, t]
            if t > 0:
                s_prev = torch.stack([act[:, t - 1]==-1, act[:, t - 1]==1]) * 1
                values[act[:,t-1]!=0, :, t] = (1 - alpha) * values[act[:,t-1]!=0, :, t-1] + alpha * torch.unsqueeze(s_prev.T[act[:,t-1]!=0], 1)
                values[act[:,t-1]==0, :, t] = values[act[:,t-1]==0, :, t-1]

        assert(torch.max(torch.abs(torch.sum(values, axis=-1) - 1)) < 1e-6)

        Rho = torch.minimum(torch.maximum(Normal(loc=torch.unsqueeze(stim, 1), scale=zetas).cdf(0), torch.tensor(1e-7)), torch.tensor(1 - 1e-7)) # pRight likelihood
        pRight, pLeft = values[:, :, :, 0] * Rho, values[:, :, :, 1] * (1 - Rho)
        pActions = torch.stack((pRight/(pRight + pLeft), pLeft/(pRight + pLeft)))

        pActions = pActions * (1 - lapses) + lapses / 2.

        p_ch     = pActions[0] * (torch.unsqueeze(act, 1) == -1) + pActions[1] * (torch.unsqueeze(act, 1) == 1) + 1 * (torch.unsqueeze(act, 1) == 0) # discard trials where agent did not answer
        p_ch     = torch.minimum(torch.maximum(p_ch, torch.tensor(1e-8)), torch.tensor(1 - 1e-8))
        logp_ch  = torch.log(p_ch)
        if return_details:
            return np.array(torch.sum(logp_ch, axis=(0, -1))), values
        return np.array(torch.sum(logp_ch, axis=(0, -1)))

        