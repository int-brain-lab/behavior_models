import numpy as np
from scipy.stats import truncnorm
import os, pickle, utils, itertools
from tqdm import tqdm
from scipy.special import logsumexp
import warnings, torch

class Model():
    '''
    This class defines the shared methods across all models
    '''
    def __init__(self, name, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side, nb_params, lb_params, ub_params, std_RW=0.02, train_method='MCMC'):
        '''
        Params:
            name (String): name of the model
            path_to_results (String): path where results will be saved
            session_uuids (array of strings): session_uuid for each session
            mouse_name (string): name of mice
            actions (nd array of size [nb_sessions, nb_trials]): actions performed by the mouse (-1/1). If the mouse did not answer 
                at one trial, pad with 0. If the sessions have different number of trials, pad the last trials with 0.
            stimuli (nd array of size [nb_sessions, nb_trials]): stimuli observed by the mouse (-1/1). If the sessions have 
                different number of trials, pad the last trials with 0.
            stim_side (nd array of size [nb_sessions, nb_trials]): stim_side of the stimuli observed by the mouse (-1/1). 
                If the sessions have different number of trials, pad the last trials with 0.
            nb_params (int): nb of parameters of the model (these parameters will be inferred)
            lb_params (array of floats): lower bounds of parameters (e.g., if `nb_params=3`, np.array([0, 0, -np.inf]))
            ub_params (array of floats): upperboud bounds of parameters (e.g., if `nb_params=3`, np.array([1, 1, np.inf]))
            train_method (string): inference method (only MCMC is possible for the moment, and forever?)
        '''
        self.name = name
        if train_method!='MCMC':
            raise NotImplementedError
        self.train_method = train_method        
        print('')
        print('Initializing {} model'.format(name))

        self.session_uuids = np.array([session_uuids[k].split('-')[0] for k in range(len(session_uuids))])
        assert(len(np.unique(self.session_uuids)) == len(self.session_uuids)), 'there is a problem in the session formatting. Contact Charles Findling'

        self.path_to_results = path_to_results
        self.lb_params, self.ub_params, self.nb_params = lb_params, ub_params, nb_params
        self.mouse_name = mouse_name
        if not os.path.exists(self.path_to_results):
            os.mkdir(self.path_to_results)
        self.path_results_mouse = self.path_to_results + self.mouse_name +'/model_{}_'.format(self.name)
        if not os.path.exists(self.path_to_results + self.mouse_name):
            os.mkdir(self.path_to_results + self.mouse_name)
        self.std_RW = std_RW

        self.actions, self.stimuli, self.stim_side = actions, stimuli, stim_side
        if (self.actions is not None) and (len(self.actions.shape)==1):
            self.actions, self.stimuli, self.stim_side = self.actions[np.newaxis], self.stimuli[np.newaxis], self.stim_side[np.newaxis]
        else:
            print('Launching in pseudo-session mode. In this mode, you only have access to the compute_prior method')

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda:0")
            print("GPU is available")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")
            print("no GPU found")

    #sessions_id = np.array([0, 1, 2], dtype=np.int); nb_chains=4; nb_steps=1000
    def mcmc(self, sessions_id, std_RW, nb_chains, nb_steps, initial_point, adaptive=True):
        '''
        Perform with inference MCMC
        Params:
            sessions_id (array of int): gives the sessions to be used for the training (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2])).        
            std_RM (float) : standard deviation of the random walk for proposal.
            nb_chains (int) : number of MCMC chains to run in parallel
            nb_steps (int) : number of MCMC steps
            initial_point (array of float of size nb_params): gives the initial_point for all chains of MCMC. All chains
                start from the same point
        '''
        if initial_point is None:
            if np.any(np.isinf(self.lb_params)) or np.any(np.isinf(self.ub_params)):
                assert(False), 'because your bounds are infinite, an initial_point must be specified'
            initial_point = (self.lb_params + self.ub_params)/2.
            import sobol_seq
            grid = np.array(sobol_seq.i4_sobol_generate(self.nb_params, nb_chains))
            initial_point = self.lb_params + grid * (self.ub_params - self.lb_params)
        
        if nb_steps is None:
            nb_steps, early_stop = int(5000), True
            if self.nb_params <= 5:
                Nburn, nb_minimum = 500, 1000
            else:
                Nburn, nb_minimum = 1000, 2000
            print('Launching MCMC procedure with {} chains, {} max steps and {} std_RW. Early stopping is activated'.format(nb_chains, nb_steps, std_RW))
        else:            
            early_stop = False
            Nburn = int(nb_steps/2)
            print('Launching MCMC procedure with {} chains, {} steps and {} std_RW'.format(nb_chains, nb_steps, std_RW))
        
        if len(initial_point.shape) == 1:
            initial_point = np.tile(initial_point[np.newaxis], (nb_chains, 1))

        if adaptive:
            print('with adaptive MCMC...')

        print('initial point for MCMC is {}'.format(initial_point))

        adaptive_proposal=None
        lkd_list = [self.evaluate(initial_point, sessions_id, clean_up_gpu_memory=False)]
        self.R_list = []
        params_list = [initial_point]
        acc_ratios = np.zeros([nb_chains])
        for i in tqdm(range(int(nb_steps))):
            if adaptive_proposal is None:
                a, b = (self.lb_params - params_list[-1]) / std_RW, (self.ub_params - params_list[-1]) / std_RW
                proposal = truncnorm.rvs(a, b, params_list[-1], std_RW)
                a_p, b_p = (self.lb_params - proposal) / std_RW, (self.ub_params - proposal) / std_RW
                prop_liks = self.evaluate(proposal, sessions_id, clean_up_gpu_memory=False)
                log_alpha = (prop_liks - lkd_list[-1] 
                            + truncnorm.logpdf(params_list[-1], a_p, b_p, proposal, std_RW).sum(axis=1)
                            - truncnorm.logpdf(proposal, a, b, params_list[-1], std_RW).sum(axis=1))
            else:
                proposal = adaptive_proposal(params_list[-1], Sigma, Lambda)
                valid = np.all((proposal > self.lb_params) * (proposal < self.ub_params), axis=1)
                proposal_modified = valid[:, np.newaxis] * proposal + (1 - valid[:, np.newaxis]) * initial_point
                prop_liks = self.evaluate(proposal_modified, sessions_id, clean_up_gpu_memory=False)
                log_alpha = (prop_liks - lkd_list[-1])
                log_alpha[valid==False] = -np.inf

            accep = np.expand_dims(log_alpha > np.log(np.random.rand(len(log_alpha))), -1)
            new_params = proposal * accep + params_list[-1] * (1 - accep)
            new_lkds   = prop_liks * np.squeeze(accep) + lkd_list[-1] * (1 - np.squeeze(accep))

            params_list.append(new_params)
            lkd_list.append(new_lkds)
            acc_ratios += np.squeeze(accep) * 1

            if early_stop and (i > Nburn) and (i > nb_minimum):
                R = self.inference_validated(np.array(params_list)[Nburn:])
                self.R_list.append(R)
                if i%100==0:
                    print('Gelman-Rubin factor is {}'.format(R))
                if np.all(np.abs(R - 1) < 0.15):
                    print('Early stopping criteria was validated at step {}. R values are: {}'.format(i, R))
                    break

            if adaptive and i>=Nburn: # Adaptive MCMC following Andrieu and Thoms 2008 or Baker 2014
                Gamma = (1/(i - Nburn + 1)**0.5)
                if i==Nburn:
                    print('Adaptive MCMC starting...')
                    from scipy.stats import multivariate_normal
                    params = np.array(params_list)[-int(Nburn/2):].reshape(-1, self.nb_params)
                    Mu = params.mean(axis=0)
                    Sigma = np.cov(params.T)
                    Lambda = np.ones(len(params_list[-1])) #(2.38**2)/self.nb_params
                    AlphaStar = 0.234
                    def adaptive_proposal(m, s, l, constrained=False):
                        list_proposals = []
                        if not constrained:
                            for k in range(len(l)):
                                list_proposals.append(multivariate_normal.rvs(mean=m[k], cov=l[k] * s))
                        else:
                            for k in range(len(l)):
                                while True:
                                    candidate = multivariate_normal.rvs(mean=m, cov=s)
                                    if np.all((candidate > self.lb_params) * (candidate < self.ub_params)):
                                        break
                                list_proposals.append(candidate)
                        return np.array(list_proposals)
                else:
                    param = params_list[-1].reshape(-1, self.nb_params)
                    Alpha_estimated = np.minimum((np.exp(log_alpha)), 1)#.mean()
                    if i%100==0:
                        print('acceptance is {}'.format(np.mean(acc_ratios/i)))
                    Lambda = Lambda * np.exp(Gamma * (Alpha_estimated - AlphaStar))
                    Mu = Mu + Gamma * (param.mean(axis=0) - Mu)
                    Sigma = Sigma + Gamma * (np.cov(param.T) - Sigma)

        print('final posterior_mean is {}'.format(np.array(params_list)[Nburn:].mean(axis=(0,1))))
        acc_ratios = acc_ratios/i
        if i==(nb_steps-1) and early_stop:
            print('Warning : inference has not converged according to Gelman-Rubin')

        if self.use_gpu: # clean up gpu memory
            torch.cuda.empty_cache()

        print('acceptance ratio is of {}. Careful, this ratio should be close to 0.234. If not, change the standard deviation of the random walk'.format(acc_ratios.mean()))
        return np.array(params_list), np.array(lkd_list), np.array(R_list)

    def inference_validated(self, parameters, method='Gelman-Rubin'):
        # implements Gelman-Rubin test, e.g., https://bookdown.org/rdpeng/advstatcomp/monitoring-convergence.html#monte-carlo-standard-errors
        if method!='Gelman-Rubin':
            return NotImplemented
        nb_samples, nb_chains, _ = parameters.shape
        chain_mean = parameters.mean(axis=0)
        post_mean = chain_mean.mean(axis=0)        
        B = nb_samples/(nb_chains - 1) * np.sum((chain_mean - post_mean)**2, axis=0)
        chain_var = np.sum((parameters - (chain_mean[np.newaxis]))**2, axis=0)/(nb_samples - 1)
        W = 1/(nb_chains) * np.sum(chain_var, axis=0)
        V = (nb_samples - 1) / nb_samples * W + 1 / nb_samples * B
        R = V/W
        return R

    def inference_successful(self, parameters, method='Gelman-Rubin'):
        R = self.inference_validated(parameters, method)
        return np.all(np.abs(R - 1) < 0.15)

    def evaluate(self, arr_params, sessions_id=None, return_details=False, clean_up_gpu_memory=True, **kwargs):
        '''
        Params:
            arr_params (nd array of size [nb_chains, nb_params]): parameters for which the likelihood will
            be computed
            sessions_id (array of int): gives the sessions on which we want to compute the likelihood (for instance, 
                if you have 4 sessions, and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
            return_details (boolean): if true returns the prior + the loglikelihood. if false, only returns the loglikelihood
        Output:
            the loglikelihood (of size [nb_sessions, nb_chains])
        '''
        if sessions_id is None and 'act' not in kwargs.keys():
            assert(False), 'session ids must be specified or explicit action/stimuli/stim_side must be passed in kwargs'
        
        if (self.actions is None) and ('act' not in kwargs.keys()): raise ValueError('No action specified')
        if (self.stimuli is None) and ('stim' not in kwargs.keys()): raise ValueError('No stimuli specified')
        if (self.stim_side is None) and ('side' not in kwargs.keys()): raise ValueError('No stim side specified')

        if (self.stim_side is not None) and (self.stim_side is not None) and (self.stim_side is not None):
            act = utils.look_up(kwargs, 'act', self.actions[sessions_id])
            stim = utils.look_up(kwargs, 'stim', self.stimuli[sessions_id])
            side = utils.look_up(kwargs, 'side', self.stim_side[sessions_id])
        else:
            if ('act' not in kwargs.keys()) or ('stim' not in kwargs.keys()) or ('side' not in kwargs.keys()):
                raise ValueError('Problem in input specification')
            else:
                act, stim, side = kwargs['act'], kwargs['stim'], kwargs['side']

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = self.compute_lkd(arr_params, act, stim, side, return_details)
        if self.use_gpu and clean_up_gpu_memory:
            torch.cuda.empty_cache()
        return res            

    def compute_lkd(arr_params, act, stim, side, return_details):
        '''
            Return the likelihood, this method must be defined in your descendant class
        '''
        return NotImplemented

    def load_or_train(self, sessions_id=None, remove_old=False, **kwargs):
        '''
        Loads the model if the model has been previously trained, otherwise trains the model
        Params:
            sessions_id (array of int): gives the sessions to be used for the training/loading (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
            remove_old (boolean): removes old saved files
        '''
        if sessions_id is None:
            sessions_id = np.arange(len(self.session_uuids))
            assert(len(self.session_uuids)==len(self.actions))
        if remove_old:
            self.remove(sessions_id, self.train_method)

        path = self.build_path(self.session_uuids[sessions_id])
        if os.path.exists(path):
            self.load(sessions_id)
            print('results found and loaded')
        else:
            self.train(sessions_id, **kwargs)

    def train(self, sessions_id, **kwargs):
        '''
        Training method
        Params:
            sessions_id (array of int): gives the sessions to be used for the training (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
        Output:
            a saved pickle file with the posterior distribution       
        '''
        if self.train_method=='MCMC':
            std_RW = utils.look_up(kwargs, 'std_RW', self.std_RW)
            nb_chains = utils.look_up(kwargs, 'nb_chains', 4)
            nb_steps = utils.look_up(kwargs, 'nb_steps', None)
            initial_point = utils.look_up(kwargs, 'initial_point', None)
            adaptive = utils.look_up(kwargs, 'adaptive', True)
            self.params_list, self.lkd_list, self.Rlist = self.mcmc(sessions_id, std_RW=std_RW, nb_chains=nb_chains, nb_steps=nb_steps, initial_point=initial_point, adaptive=adaptive)
            path = self.build_path(self.session_uuids[sessions_id])
            pickle.dump([self.params_list, self.lkd_list, self.Rlist], open(path, 'wb'))
            print('results of inference SAVED')
        else:
            return NotImplemented

    def load(self, sessions_id):
        '''
        Load method. This method should not be called directly. Call the load_or_train() method instead
        Params:
            sessions_id (array of int): gives the sessions to be used for the loading (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
        Ouput:
            Loads an existing file.         
        '''
        if self.train_method=='MCMC':
            path = self.build_path(self.session_uuids[sessions_id])
            try:
                [self.params_list, self.lkd_list, self.Rlist] = pickle.load(open(path, 'rb'))
            except:
                [self.params_list, self.lkd_list] = pickle.load(open(path, 'rb')) # to take out on longer term <- necessary for version continuity
                pass
        else:
            return NotImplemented

    def remove(self, sessions_id):
        '''
        Remove method. This method removes the past saved results.
        Params:
            sessions_id (array of int): gives the sessions used for the training (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
        Ouput:
            Removes the previously found results
        '''
        path = self.build_path(self.session_uuids[sessions_id])
        if os.path.exists(path):
            os.remove(path)
            print('results deleted')
        else:
            print('no results were saved')

    # act=self.actions; stim=self.stimuli; side=self.stim_side
    def compute_prior(self, act=None, stim=None, side=None, sessions_id=None, parameter_type='whole_posterior'):
        '''
        Compute_prior method.
        Params:
            act (nd array of size [nb_sessions, nb_trials]): actions performed by the mouse (-1/1). If the mouse did not answer 
                at one trial, pad with 0. If the sessions have different number of trials, pad the last trials with 0.
            stim (nd array of size [nb_sessions, nb_trials]): stimuli observed by the mouse (-1/1). If the sessions have 
                different number of trials, pad the last trials with 0.
            side (nd array of size [nb_sessions, nb_trials]): stim_side of the stimuli observed by the mouse (-1/1). 
                If the sessions have different number of trials, pad the last trials with 0.
            sessions_id (array of int): gives the sessions used for the traning (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
            parameter_type (string) : how the prior is computed wrt the parameters. 'posterior_mean' and 'maximum_a_posteriori' are available
        Ouput:
            Computes the prior and accuracy for given act/stim/side   
        '''
        if (act is None) and (self.actions is None):
            print('no actions is specified')
        if (stim is None) and (self.stimuli is None):
            print('no stimuli is specified')
        if (side is None) and (self.stim_side is None):
            print('no stim_side is specified')

        if act is None: act = self.actions
        if stim is None: stim = self.stimuli
        if side is None: side = self.stim_side

        if (not hasattr(self, 'params_list')):
            if sessions_id is None:
                sessions_id = np.arange(len(self.session_uuids))
            path = self.build_path(self.session_uuids[sessions_id])
            if os.path.exists(path):
                self.load(sessions_id)
            else:
                raise ValueError('the model has not be trained')

        if self.train_method=='MCMC': 
            assert(parameter_type in ['MAP', 'posterior_mean', 'whole_posterior']), 'parameter_type must be MAP, posterior_mean or whole_posterior'
        if self.train_method!='MCMC':
            return NotImplemented

        if parameter_type=='posterior_mean':
            print('Using posterior mean')
            nb_steps = len(self.params_list)
            parameters_chosen = self.params_list[-500:].mean(axis=(0,1))[np.newaxis]
        elif parameter_type=='maximum_a_posteriori':
            print('Using MAP')
            xmax, ymax = np.where(self.lkd_list==np.max(self.lkd_list))
            parameters_chosen = self.params_list[xmax[0], ymax[0]][np.newaxis]
        elif parameter_type=='whole_posterior':
            print('Using whole posterior')
            parameters_chosen = self.params_list[-500:].reshape(-1, self.nb_params)
        if len(act.shape)==1:
            act, stim, side = act[np.newaxis], stim[np.newaxis], side[np.newaxis]
        loglkd, priors = self.evaluate(parameters_chosen, return_details=True, act=act, stim=stim, side=side)
        llk = logsumexp(loglkd) - np.log(len(loglkd))
        accuracy = np.exp(llk/np.sum(act!=0))
        return np.squeeze(np.mean(np.array(priors), axis=1)), llk, accuracy

    def score(self, sessions_id_test, sessions_id, parameter_type='whole_posterior', remove_old=False, param=None):
        '''
        Scores the model on session_id. NB: to implement cross validation, do not train and test on the same sessions
        Params:
            sessions_id (array of int): gives the sessions used for the training (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
            sessions_id_test (array of int): gives the sessions on which we want to test             
            parameter_type (string) : how the prior is computed wrt the parameters. 'posterior_mean' and 'maximum_a_posteriori' are available
        Outputs:
            accuracy and loglkd on new sessions
        '''
        path = self.build_path(self.session_uuids[sessions_id], self.session_uuids[sessions_id_test])
        if (remove_old or param is not None) and os.path.exists(path):
            os.remove(path)
        elif os.path.exists(path):
            [prior, loglkd, accuracy] = pickle.load(open(path, 'rb'))
            print('saved score results found')
            print('accuracy on test sessions: {}'.format(accuracy))
            return loglkd, accuracy

        self.load(sessions_id)
        act, stim, side = self.actions[sessions_id_test], self.stimuli[sessions_id_test], self.stim_side[sessions_id_test]

        if param is not None:
            print('custom parameters: {}'.format(param))
            loglkd, priors = self.evaluate(param[np.newaxis], return_details=True, act=act, stim=stim, side=side)
            accuracy = np.exp(loglkd/np.sum(act!=0))
            print('accuracy on test sessions: {}'.format(accuracy))
            return loglkd, accuracy
        else:
            prior, loglkd, accuracy = self.compute_prior(act, stim, side, sessions_id=sessions_id, parameter_type=parameter_type)
            pickle.dump([prior, loglkd, accuracy], open(path, 'wb'))
            print('accuracy on test sessions: {}'.format(accuracy))
        return loglkd, accuracy

    def build_path(self, l_sessionuuids_train, l_sessionuuids_test=None):
        '''
        Generates the path where the results will be saved
        Params:
            l_sessionuuids_train (array of int)
            l_sessionuuids_test (array of int)
        Ouput:
            formatted path where the results are saved
        '''        
        str_sessionuuids = ''
        for k in range(len(l_sessionuuids_train)): str_sessionuuids += '_sess{}_{}'.format(k+1, l_sessionuuids_train[k])
        if l_sessionuuids_test is None:
            path = self.path_results_mouse + 'train_{}_train{}.pkl'.format(self.train_method, str_sessionuuids)
            return path
        else:
            str_sessionuuids_test = ''
            for k in range(len(l_sessionuuids_test)): str_sessionuuids_test += '_sess{}_{}'.format(k+1, l_sessionuuids_test[k])
            path = self.path_results_mouse + 'train_{}_train{}_test{}.pkl'.format(self.train_method, str_sessionuuids, str_sessionuuids_test)
            return path

    # act=self.actions; stim=self.stimuli; side=self.stim_side
    def get_parameters(self, sessions_id=None, parameter_type='all'):
        '''
        get parameters method.
        Params:
            sessions_id (array of int): gives the sessions on which the training took place (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
            parameter_type (string) : how the prior is computed wrt the parameters. 'posterior_mean', 'maximum_a_posteriori' and 'all' are available
        '''        
        if not hasattr(self, 'params_list'):
            if sessions_id is None:
                sessions_id = np.arange(len(self.actions))
            path = self.build_path(self.session_uuids[sessions_id])
            if os.path.exists(path):
                self.load(sessions_id)
            else:
                print('call the load_or_train() function')

        if self.train_method=='MCMC': 
            assert(parameter_type in ['maximum_a_posteriori', 'posterior_mean', 'all']), 'parameter_type must be maximum_a_posteriori, posterior_mean or all'
        if self.train_method!='MCMC':
            return NotImplemented

        if parameter_type=='posterior_mean':
            nb_steps = len(self.params_list)
            parameters_chosen = self.params_list[-int(nb_steps/2):].mean(axis=(0,1)) # int(nb_steps/2)
            return parameters_chosen
        elif parameter_type=='maximum_a_posteriori':
            xmax, ymax = np.where(self.lkd_list==np.max(self.lkd_list))
            parameters_chosen = self.params_list[xmax[0], ymax[0]]
            return parameters_chosen
        else:
            return self.params_list
        

