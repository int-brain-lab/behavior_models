from models import model
import torch, utils
import numpy as np
from torch.distributions.normal import Normal

unsqueeze = lambda x : torch.unsqueeze(torch.unsqueeze(x, 0), -1)

class expSmoothing_stimside_SE(model.Model):
    '''
        Model where the prior is based on an exponential estimation of the previous stimulus side
    '''

    def __init__(self, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side, stimulated):
        name = 'expSmoothingStimSide_SE'
        nb_params, lb_params, ub_params = 6, np.array([0, 0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, .5, .5])
        std_RW = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
        self.stimulated = stimulated
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
        alpha_unstimulated, alpha_stimulated, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(arr_params).T      
        loglikelihood = np.zeros(nb_chains)
        act, stim, side = torch.tensor(act), torch.tensor(stim), torch.tensor(side)
        nb_sessions = len(act)

        values = torch.zeros([nb_sessions, nb_chains, act.shape[-1], 2], dtype=torch.float64) + 0.5

        alpha = torch.unsqueeze(unsqueeze(alpha_stimulated) * (torch.unsqueeze(self.stimulated,1)==1) + unsqueeze(alpha_unstimulated) * (torch.unsqueeze(self.stimulated,1)==0), -1)
        zetas = unsqueeze(zeta_pos) * (torch.unsqueeze(side,1) > 0) + unsqueeze(zeta_neg) * (torch.unsqueeze(side,1) <= 0)
        lapses = unsqueeze(lapse_pos) * (torch.unsqueeze(side,1) > 0) + unsqueeze(lapse_neg) * (torch.unsqueeze(side,1) <= 0)

        for t in range(act.shape[-1]):
            if t > 0:
                a_prev = torch.stack([side[:, t - 1]==-1, side[:, t - 1]==1]) * 1
                values[:, :, t] = (1 - alpha[:, :, t]) * values[:, :, t-1] + alpha[:, :, t] * torch.unsqueeze(a_prev.T, 1)
                values[act[:,t-1]==0, :, t] = values[act[:,t-1]==0, :, t-1]

        assert(torch.max(torch.abs(torch.sum(values, axis=-1) - 1)) < 1e-6)

        Rho = torch.minimum(torch.maximum(Normal(loc=torch.unsqueeze(stim, 1), scale=zetas).cdf(torch.tensor(0)), torch.tensor(1e-7)), torch.tensor(1 - 1e-7)) # pRight likelihood
        pRight, pLeft = values[:, :, :, 0] * Rho, values[:, :, :, 1] * (1 - Rho)
        pActions = torch.stack((pRight/(pRight + pLeft), pLeft/(pRight + pLeft)))

        belief     = pActions[0] * (torch.unsqueeze(act, 1) == -1) + pActions[1] * (torch.unsqueeze(act, 1) == 1)
        correct    = (act == side) * 1 
        prediction_error = (torch.unsqueeze(correct, 1) - belief)

        pActions = pActions * (1 - lapses) + lapses / 2.

        p_ch     = pActions[0] * (torch.unsqueeze(act, 1) == -1) + pActions[1] * (torch.unsqueeze(act, 1) == 1) + 1 * (torch.unsqueeze(act, 1) == 0) # discard trials where agent did not answer
        p_ch     = torch.minimum(torch.maximum(p_ch, torch.tensor(1e-8)), torch.tensor(1 - 1e-8))
        logp_ch  = torch.log(p_ch)
        if return_details:
            return logp_ch, values[:, :, :, 1], prediction_error
        return np.array(torch.sum(logp_ch, axis=(0, -1)))
