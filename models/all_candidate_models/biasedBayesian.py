from models import model, utils
import torch
import numpy as np
from torch.distributions.normal import Normal

unsqueeze = lambda x : torch.unsqueeze(torch.unsqueeze(x, 0), -1)

class biased_Bayesian(model.Model):
    '''
        Model where the prior is based on an exponential estimation of the previous stimulus side
    '''

    def __init__(self, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side, repetition_bias=False, loguniform_prior=False, UB_fit=False):
        name = 'biased_bayesian' + '_with_repBias' * repetition_bias + '_with_logPriors' * loguniform_prior + '_UB_fitted' * (UB_fit == False)
        if UB_fit and not loguniform_prior:
            nb_params, lb_params, ub_params = 8, np.array([0, 0, 0, 0.5, 0, 0, 0, 0]), np.array([40, 50, 50, 1, 1, 1, .5, .5])
            std_RW = np.array([.5, .5, 1, 0.01, 0.01, 0.01, 0.01, 0.01])
        if not UB_fit and not loguniform_prior:
            nb_params, lb_params, ub_params = 7, np.array([0, 0, 0.5, 0, 0, 0, 0]), np.array([40, 60, 1, 1, 1, .5, .5])
            std_RW = np.array([1, 2, 0.01, 0.02, 0.02, 0.01, 0.01])            
        elif not UB_fit and loguniform_prior:
            # LB ~ logU([1, 40]) and  tau ~ logU([2, 100])
            nb_params, lb_params, ub_params = 7, np.array([0, 0, 0.5, 0, 0, 0, 0]), np.array([np.log(40), np.log(100), 1, 1, 1, .5, .5])
            std_RW = np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.01, 0.01])
        else:
            raise ValueError('model instance not supported')
        self.loguniform_prior, self.UB_fit = loguniform_prior, UB_fit
        self.repetition_bias = repetition_bias
        if repetition_bias:
            nb_params += 1
            lb_params, ub_params = np.append(lb_params, 0), np.append(ub_params, .5)
            std_RW = np.concatenate((std_RW, np.array([0.01])))
        super().__init__(name, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side, nb_params, lb_params, ub_params, std_RW)
        self.nb_typeblocks = 3
        if self.UB_fit:
            self.nb_blocklengths = int(ub_params[:3].sum())
        else:
            self.nb_blocklengths = 100

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
        if not self.repetition_bias and self.UB_fit and not self.loguniform_prior:
            tau0, tau1, tau2, gamma, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(arr_params, device=self.device, dtype=torch.float32).T
            lb, tau, ub = tau0, tau0 + tau1, tau0 + tau1 + tau2
        elif self.UB_fit and self.repetition_bias and not self.loguniform_prior:
            tau0, tau1, tau2, gamma, zeta_pos, zeta_neg, lapse_pos, lapse_neg, rep_bias = torch.tensor(arr_params, device=self.device, dtype=torch.float32).T
            lb, tau, ub = tau0, tau0 + tau1, tau0 + tau1 + tau2
        elif not self.UB_fit and not self.loguniform_prior and not self.repetition_bias:
            tau0, tau1, gamma, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(arr_params, device=self.device, dtype=torch.float32).T
            lb, tau, ub = tau0, tau0 + tau1, torch.zeros(len(gamma), device=self.device, dtype=torch.float32) + self.nb_blocklengths
        elif not self.UB_fit and self.loguniform_prior and self.repetition_bias:
            loglb, logtau, gamma, zeta_pos, zeta_neg, lapse_pos, lapse_neg, rep_bias = torch.tensor(arr_params, device=self.device, dtype=torch.float32).T
            lb, tau, ub = torch.exp(loglb), torch.exp(logtau), torch.zeros(len(gamma), device=self.device, dtype=torch.float32) + self.nb_blocklengths            
        elif not self.UB_fit and self.loguniform_prior and not self.repetition_bias:
            loglb, logtau, gamma, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(arr_params, device=self.device, dtype=torch.float32).T
            lb, tau, ub = torch.exp(loglb), torch.exp(logtau), torch.zeros(len(gamma), device=self.device, dtype=torch.float32) + self.nb_blocklengths
        else:
            raise ValueError('model instance not supported')
        act, stim, side = torch.tensor(act, device=self.device, dtype=torch.float32), torch.tensor(stim, device=self.device, dtype=torch.float32), torch.tensor(side, device=self.device, dtype=torch.float32)
        nb_sessions = len(act)

        alpha = torch.zeros([nb_sessions, nb_chains, act.shape[-1], self.nb_blocklengths, self.nb_typeblocks], device=self.device, dtype=torch.float32)
        alpha[:, :, 0, 0, 1] = 1
        alpha = alpha.reshape(nb_sessions, nb_chains, -1, self.nb_typeblocks * self.nb_blocklengths)
        h = torch.zeros([nb_sessions, nb_chains, self.nb_typeblocks * self.nb_blocklengths], device=self.device, dtype=torch.float32)

        zetas = unsqueeze(zeta_pos) * (torch.unsqueeze(side,1) > 0) + unsqueeze(zeta_neg) * (torch.unsqueeze(side,1) <= 0)
        lapses = unsqueeze(lapse_pos) * (torch.unsqueeze(side,1) > 0) + unsqueeze(lapse_neg) * (torch.unsqueeze(side,1) <= 0)

        # build transition matrix
        b = torch.zeros([self.nb_blocklengths, 3, 3], device=self.device, dtype=torch.float32)
        b[1:][:,0,0], b[1:][:,1,1], b[1:][:,2,2] = 1, 1, 1 # case when l_t > 0
        b[0][0][-1], b[0][-1][0], b[0][1][np.array([0, 2])] = 1, 1, 1./2 # case when l_t = 1
        n = torch.unsqueeze(torch.arange(1, self.nb_blocklengths+1, device=self.device, dtype=torch.float32), 0)
        ref    = torch.exp(-n/torch.unsqueeze(tau, -1)) * (torch.unsqueeze(lb, -1) <= n) * (torch.unsqueeze(ub, -1) >= n)
        hazard = torch.cummax(ref/torch.flip(torch.cumsum(torch.flip(ref, (1,)), 1) + 1e-18, (1,)), 1)[0]
        padding = torch.unsqueeze(torch.zeros(self.nb_blocklengths-1, device=self.device, dtype=torch.float32), 0)

        l = torch.stack([torch.tensor(torch.cat((torch.unsqueeze(hazard[i], -1), torch.cat(
                        (torch.diag(1 - hazard[i, :-1]), padding), axis=0)), axis=-1), dtype=torch.float32) for i in range(len(hazard))]) # l_{t-1}, l_t

        transition = torch.stack([1e-12 + torch.transpose(l[k][:,:,np.newaxis,np.newaxis] * b[np.newaxis], 1, 2).reshape(self.nb_typeblocks * self.nb_blocklengths, -1) for k in range(len(l))])        
        # likelihood
        Rhos = Normal(loc=torch.unsqueeze(stim, 1), scale=zetas).cdf(0)
        ones = torch.ones((nb_chains, nb_sessions, act.shape[-1]) , device=self.device, dtype=torch.float32)
        gamma_unsqueezed, side_unsqueezed = torch.unsqueeze(torch.unsqueeze(gamma, 1), -1), torch.unsqueeze(side, 0)        
        lks = torch.stack([gamma_unsqueezed*(side_unsqueezed==-1) + (1-gamma_unsqueezed) * (side_unsqueezed==1), ones * 1./2, gamma_unsqueezed*(side_unsqueezed==1) + (1-gamma_unsqueezed)*(side_unsqueezed==-1)]).T
        to_update = torch.unsqueeze(torch.unsqueeze(act!=0, -1), -1) * 1

        for i_trial in range(act.shape[-1]):
            if i_trial > 0:
                alpha[:, :, i_trial] = torch.sum(torch.unsqueeze(h, -1) * transition, axis=2) * to_update[:,i_trial-1] + alpha[:,:,i_trial-1] * (1 - to_update[:,i_trial-1])
            h = alpha[:, :, i_trial] * lks[i_trial].repeat(1, 1, self.nb_blocklengths)
            h = h/torch.unsqueeze(torch.sum(h, axis=-1), -1)

        predictive = torch.sum(alpha.reshape(nb_sessions, nb_chains, -1, self.nb_blocklengths, self.nb_typeblocks), 3)
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

        