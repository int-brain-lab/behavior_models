from models import model, utils
import torch
import numpy as np
from torch.distributions.normal import Normal

unsqueeze = lambda x : torch.unsqueeze(torch.unsqueeze(x, 0), -1)

class biased_ApproxBayesian(model.Model):
    '''
        Model where the prior is based on an exponential estimation of the previous stimulus side
    '''

    def __init__(self, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side, repetition_bias=False, random_initial_block=False):
        name = 'biased_Approxbayesian' + '_random_initial_block' * random_initial_block + '_with_repBias' * repetition_bias
        nb_params, lb_params, ub_params = 6, np.array([0, 0.5, 0, 0, 0, 0]), np.array([.5, 1, 1, 1, .5, .5])
        std_RW = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
        self.repetition_bias = repetition_bias
        self.random_initial_block = random_initial_block
        if repetition_bias:
            nb_params += 1
            lb_params, ub_params = np.append(lb_params, 0), np.append(ub_params, .5)
            std_RW = np.array([.02, 0.02, 0.05, 0.05, 0.01, 0.01, 0.01])
        super().__init__(name, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side, nb_params, lb_params, ub_params, std_RW)
        self.nb_typeblocks = 3

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
            prior (array of shape [nb_sessions, nb_chains, nb_trials]): prior for each chain and session
        '''
        nb_chains = len(arr_params)
        if not self.repetition_bias:
            vol, gamma, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(arr_params, device=self.device, dtype=torch.float32).T
        else:
            vol, gamma, zeta_pos, zeta_neg, lapse_pos, lapse_neg, rep_bias = torch.tensor(arr_params, device=self.device, dtype=torch.float32).T
        act, stim, side = torch.tensor(act, device=self.device, dtype=torch.float32), torch.tensor(stim, device=self.device, dtype=torch.float32), torch.tensor(side, device=self.device, dtype=torch.float32)
        nb_sessions = len(act)

        predictive = torch.zeros([nb_sessions, nb_chains, act.shape[-1], self.nb_typeblocks], device=self.device, dtype=torch.float32)
        if self.random_initial_block:
            predictive[:, :, 0, :] = 1./self.nb_typeblocks
        else:
            predictive[:, :, 0, 1] = 1
        h = torch.zeros([nb_sessions, nb_chains, self.nb_typeblocks], device=self.device, dtype=torch.float32)

        zetas = unsqueeze(zeta_pos) * (torch.unsqueeze(side,1) > 0) + unsqueeze(zeta_neg) * (torch.unsqueeze(side,1) <= 0)
        lapses = unsqueeze(lapse_pos) * (torch.unsqueeze(side,1) > 0) + unsqueeze(lapse_neg) * (torch.unsqueeze(side,1) <= 0)

        # build transition matrix
        transition = torch.zeros([len(vol), self.nb_typeblocks, self.nb_typeblocks], device=self.device, dtype=torch.float32)
        zeros = torch.zeros(len(vol), dtype = torch.float32, device=self.device)
        transition[:, 0] = torch.stack([1 - vol, zeros, vol]).T
        transition[:, 1] = torch.stack([vol/2, 1 - vol, vol/2]).T
        transition[:, 2] = torch.stack([vol, zeros, 1 - vol]).T

        # likelihood
        Rhos = Normal(loc=torch.unsqueeze(stim, 1), scale=zetas).cdf(0)
        ones = torch.ones((nb_chains, nb_sessions, act.shape[-1]) , device=self.device, dtype=torch.float32)
        gamma_unsqueezed, side_unsqueezed = torch.unsqueeze(torch.unsqueeze(gamma, 1), -1), torch.unsqueeze(side, 0)        
        lks = torch.stack([gamma_unsqueezed*(side_unsqueezed==-1) + (1-gamma_unsqueezed) * (side_unsqueezed==1), ones * 1./2, gamma_unsqueezed*(side_unsqueezed==1) + (1-gamma_unsqueezed)*(side_unsqueezed==-1)]).T
        to_update = torch.unsqueeze(torch.unsqueeze(act!=0, -1), -1) * 1

        for i_trial in range(act.shape[-1]):
            if i_trial > 0:
                predictive[:, :, i_trial] = torch.sum(torch.unsqueeze(h, -1) * transition, axis=2) * to_update[:,i_trial-1] + predictive[:,:,i_trial-1] * (1 - to_update[:,i_trial-1])
            h = predictive[:, :, i_trial] * lks[i_trial]
            h = h/torch.unsqueeze(torch.sum(h, axis=-1), -1)

        #predictive = torch.sum(alpha.reshape(nb_sessions, nb_chains, -1, self.nb_blocklengths, self.nb_typeblocks), 3)
        Pis  = predictive[:, :, :, 0] * unsqueeze(gamma) + predictive[:, :, :, 1] * 0.5 + predictive[:, :, :, 2] * (1 - unsqueeze(gamma))
        pRight, pLeft = Pis * Rhos, (1 - Pis) * (1 - Rhos)
        pActions = torch.stack((pRight/(pRight + pLeft), pLeft/(pRight + pLeft)))

        unsqueezed_lapses = torch.unsqueeze(lapses, 0)
        if self.repetition_bias:
            unsqueezed_rep_bias = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(rep_bias, 0), 0), -1)
            pActions[:,:,:,0] = pActions[:,:,:,0] * (1 - unsqueezed_lapses[:,:,:,0]) + unsqueezed_lapses[:,:,:,0] / 2.
            pActions[:,:,:,1:] = pActions[:,:,:,1:] * (1 - unsqueezed_lapses[:,:,:,1:] - unsqueezed_rep_bias) + unsqueezed_lapses[:,:,:,1:] / 2. + unsqueezed_rep_bias * torch.unsqueeze(torch.stack(((act[:,:-1]==-1) * 1, (act[:,:-1]==1) * 1)), 2)
        else:
            pActions = pActions * (1 - torch.unsqueeze(lapses, 0)) + torch.unsqueeze(lapses, 0) / 2.

        pActions[torch.isnan(pActions)] = 0

        p_ch     = pActions[0] * (torch.unsqueeze(act, 1) == -1) + pActions[1] * (torch.unsqueeze(act, 1) == 1) + 1 * (torch.unsqueeze(act, 1) == 0) # discard trials where agent did not answer

        priors   = 1 - torch.tensor(Pis.detach(), device='cpu')
        p_ch_cpu = torch.tensor(p_ch.detach(), device='cpu')    
        logp_ch = torch.log(torch.minimum(torch.maximum(p_ch_cpu, torch.tensor(1e-8)), torch.tensor(1 - 1e-8)))

        # clean up gpu memory
        # if self.use_gpu:
        #     del tau0, tau1, tau2, gamma, zeta_pos, zeta_neg, lapse_pos, lapse_neg, lb, tau, ub, act, stim, side, s, lks
        #     del alpha, h, zetas, lapses, b, n, ref, hazard, padding, l, transition, ones, Rhos, gamma_unsqueezed
        #     del predictive, Pis, pRight, pLeft, pActions, p_ch, unsqueezed_lapses
        #     if self.repetition_bias:
        #         del rep_bias, unsqueezed_rep_bias
        #     torch.cuda.empty_cache()

        if return_details:
            return logp_ch, priors
        return np.array(torch.sum(logp_ch, axis=(0, -1)))

        