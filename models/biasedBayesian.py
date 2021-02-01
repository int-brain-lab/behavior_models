from models import model
import torch, utils
import numpy as np
from torch.distributions.normal import Normal

unsqueeze = lambda x : torch.unsqueeze(torch.unsqueeze(x, 0), -1)

class biased_Bayesian(model.Model):
    '''
        Model where the prior is based on an exponential estimation of the previous stimulus side
    '''

    def __init__(self, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side):
        name = 'biased_bayesian'
        nb_params, lb_params, ub_params = 8, np.array([0, 0, 0, 0.5, 0, 0, 0, 0]), np.array([40, 50, 50, 1, 1, 1, .5, .5])
        std_RW = np.array([.5, .5, 1, 0.01, 0.01, 0.01, 0.01, 0.01])
        initial_point = np.array([20, 40, 40, 0.8, 0.5, 0.5, 0.1, 0.1])
        super().__init__(name, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side, nb_params, lb_params, ub_params, std_RW, initial_point)
        self.nb_blocklengths, self.nb_typeblocks = int(ub_params[:3].sum()), 3
        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")
            print("Running on the CPU")

    def compute_lkd(self, arr_params, act, stim, side, return_details):
        nb_chains = len(arr_params)
        tau0, tau1, tau2, gamma, zeta_pos, zeta_neg, lapse_pos, lapse_neg = torch.tensor(arr_params, device=self.device).T
        act, stim, side = torch.tensor(act, device=self.device), torch.tensor(stim, device=self.device), torch.tensor(side, device=self.device)
        nb_sessions = len(act)
        lb, tau, ub = tau0, tau0 + tau1, tau0 + tau1 + tau2

        alpha = torch.zeros([nb_sessions, nb_chains, self.nb_trials, self.nb_blocklengths, self.nb_typeblocks], device=self.device, dtype=torch.float64)
        alpha[:, :, 0, 0, 1] = 1
        alpha = alpha.reshape(nb_sessions, nb_chains, -1, self.nb_typeblocks * self.nb_blocklengths)
        h = torch.zeros([nb_sessions, nb_chains, self.nb_typeblocks * self.nb_blocklengths], device=self.device)

        zetas = unsqueeze(zeta_pos) * (torch.unsqueeze(side,1) > 0) + unsqueeze(zeta_neg) * (torch.unsqueeze(side,1) <= 0)
        lapses = unsqueeze(lapse_pos) * (torch.unsqueeze(side,1) > 0) + unsqueeze(lapse_neg) * (torch.unsqueeze(side,1) <= 0)

        # build transition matrix
        b = torch.zeros([self.nb_blocklengths, 3, 3], device=self.device)
        b[1:][:,0,0], b[1:][:,1,1], b[1:][:,2,2] = 1, 1, 1 # case when l_t > 0
        b[0][0][-1], b[0][-1][0], b[0][1][np.array([0, 2])] = 1, 1, 1./2 # case when l_t = 1
        n = torch.unsqueeze(torch.arange(1, self.nb_blocklengths+1, device=self.device), 0)
        ref    = torch.exp(-n/torch.unsqueeze(tau, -1)) * (torch.unsqueeze(lb, -1) <= n) * (torch.unsqueeze(ub, -1) >= n)
        hazard = torch.cummax(ref/torch.flip(torch.cumsum(torch.flip(ref, (1,)), 1) + 1e-18, (1,)), 1)[0]
        padding = torch.unsqueeze(torch.zeros(self.nb_blocklengths-1, device=self.device), 0)
        # l = torch.cat((torch.unsqueeze(hazard[0], -1), torch.cat(
        #             (torch.diag(1 - hazard[0, :-1]), padding), axis=0)), axis=-1) # l_{t-1}, l_t

        l = torch.stack([torch.cat((torch.unsqueeze(hazard[i], -1), torch.cat(
                        (torch.diag(1 - hazard[i, :-1]), padding), axis=0)), axis=-1) for i in range(len(hazard))]) # l_{t-1}, l_t

        transition = torch.stack([1e-12 + torch.transpose(l[k][:,:,np.newaxis,np.newaxis] * b[np.newaxis], 1, 2).reshape(self.nb_typeblocks * self.nb_blocklengths, -1) for k in range(len(l))])
        ones = torch.ones((nb_chains, nb_sessions) , device=self.device)
        # likelihood
        Rhos = Normal(loc=torch.unsqueeze(stim, 1), scale=zetas).cdf(0)
        gamma_unsqueezed = torch.unsqueeze(gamma, 1)

        for i_trial in range(self.nb_trials):
            s = torch.unsqueeze(side[:, i_trial], 0)
            lks = torch.stack([gamma_unsqueezed*(s==-1) + (1-gamma_unsqueezed) * (s==1), ones * 1./2, gamma_unsqueezed*(s==1) + (1-gamma_unsqueezed)*(s==-1)]).T

            # save priors
            if i_trial > 0:
                alpha[act[:, i_trial-1]!=0, :, i_trial] = torch.sum(torch.unsqueeze(h, -1) * transition, axis=2)[act[:, i_trial-1]!=0] #torch.logsumexp(h[:, :, :, np.newaxis] + self.t, axis=(2))
                alpha[act[:, i_trial-1]==0, :, i_trial] = alpha[act[:, i_trial-1]==0, :, (i_trial-1)]
            h = alpha[:, :, i_trial] * lks.repeat(1, 1, self.nb_blocklengths)
            h = h/torch.unsqueeze(torch.sum(h, axis=-1), -1)

        predictive = torch.sum(alpha.reshape(nb_sessions, nb_chains, -1, self.nb_blocklengths, self.nb_typeblocks), 3)
        Pis  = predictive[:, :, :, 0] * unsqueeze(gamma) + predictive[:, :, :, 1] * 0.5 + predictive[:, :, :, 2] * (1 - unsqueeze(gamma))
        pRight, pLeft = Pis * Rhos, (1 - Pis) * (1 - Rhos)
        pActions = torch.stack((pRight/(pRight + pLeft), pLeft/(pRight + pLeft)))
        pActions = pActions * (1 - lapses) + lapses / 2.
        pActions[torch.isnan(pActions)] = 0

        p_ch     = pActions[0] * (torch.unsqueeze(act, 1) == -1) + pActions[1] * (torch.unsqueeze(act, 1) == 1) + 1 * (torch.unsqueeze(act, 1) == 0) # discard trials where agent did not answer
        logp_ch = torch.log(torch.minimum(torch.maximum(p_ch.cpu(), torch.tensor(1e-8)), torch.tensor(1 - 1e-8)))

        # clean up gpu memory
        # if self.use_gpu:
        #     del zeta_pos, zeta_neg, lapse_pos, lapse_neg, lks, ones
        #     del act, stim, side, priors, h, alpha, zetas, lapses
        #     del b, n, ref, hazard, padding, l, transition, Rhos
        #     del predictive, Pis, pRight, pLeft, pActions, p_ch

        if return_details:
            return np.array(torch.sum(logp_ch, axis=(0, -1))), Pis
        return np.array(torch.sum(logp_ch, axis=(0, -1)))

        