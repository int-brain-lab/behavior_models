import numpy as np
from scipy.stats import truncnorm
import os, pickle, utils
from tqdm import tqdm

class Model():
    '''
    This class defines the shared methods across all models
    '''
    def __init__(self, name, path_to_results, session_uuids, mouse_name, actions, stimuli, stim_side, nb_params, lb_params, ub_params):
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
        '''
        self.name = name
        self.path_to_results = path_to_results        
        self.actions, self.stimuli, self.stim_side = actions, stimuli, stim_side
        if len(self.actions.shape)==1:
            self.actions, self.stimuli, self.stim_side = self.actions[np.newaxis], self.stimuli[np.newaxis], self.stim_side[np.newaxis]        
        self.lb_params, self.ub_params, self.nb_params = lb_params, ub_params, nb_params
        self.nb_trials = self.actions.shape[-1]
        self.session_uuids = np.array([session_uuids[k].split('-')[0] for k in range(len(session_uuids))])
        assert(len(np.unique(self.session_uuids)) == len(self.session_uuids)), 'there is a problem in the session formatting. Contact Charles Findling'
        self.mouse_name = mouse_name
        if not os.path.exists(self.path_to_results):
            os.mkdir(self.path_to_results)
        self.path_results_mouse = self.path_to_results + self.mouse_name +'/model_{}_'.format(self.name)
        if not os.path.exists(self.path_to_results + self.mouse_name):
            os.mkdir(self.path_to_results + self.mouse_name)

    #sessions_id = np.array([0, 1, 2], dtype=np.int); nb_chains=4; nb_steps=1000
    def mcmc(self, sessions_id, std_RW, nb_chains, nb_steps, initial_point):
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
        
        initial_point = np.tile(initial_point[np.newaxis], (nb_chains, 1))

        lkd_list = [self.evaluate(initial_point, sessions_id)]
        params_list = [initial_point]
        acc_ratios = np.zeros([nb_chains])
        for i in tqdm(range(nb_steps)):
            a, b = (self.lb_params - params_list[-1]) / std_RW, (self.ub_params - params_list[-1]) / std_RW
            proposal = truncnorm.rvs(a, b, params_list[-1], std_RW)
            a_p, b_p = (self.lb_params - proposal) / std_RW, (self.ub_params - proposal) / std_RW
            prop_liks = self.evaluate(proposal, sessions_id)
            log_alpha = (prop_liks - lkd_list[-1] 
                        + truncnorm.logpdf(params_list[-1], a_p, b_p, proposal, std_RW).sum(axis=1)
                        - truncnorm.logpdf(proposal, a, b, params_list[-1], std_RW).sum(axis=1))
            accep = np.expand_dims(log_alpha > np.log(np.random.rand(len(log_alpha))), -1)
            new_params = proposal * accep + params_list[-1] * (1 - accep)
            new_lkds   = prop_liks * np.squeeze(accep) + lkd_list[-1] * (1 - np.squeeze(accep))

            params_list.append(new_params)
            lkd_list.append(new_lkds)
            acc_ratios += np.squeeze(accep) * 1
        acc_ratios = acc_ratios/nb_steps
        print('acceptance ratio is of {}. Careful, this ratio should be close to 0.15. If not, change the standard deviation of the random walk'.format(acc_ratios.mean()))
        return np.array(params_list), np.array(lkd_list)

    def evaluate(self, arr_params, sessions_id, return_details=False, **kwargs):
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
        act = utils.look_up(kwargs, 'act', self.actions[sessions_id])
        stim = utils.look_up(kwargs, 'stim', self.stimuli[sessions_id])
        side = utils.look_up(kwargs, 'side', self.stim_side[sessions_id])
        return self.compute_lkd(arr_params, act, stim, side, return_details)

    def compute_lkd(arr_params, act, stim, side, return_details):
        '''
            Return the likelihood, this method must be defined in your descendant class
        '''
        return NotImplemented

    def load_or_train(self, sessions_id=None, train_method='MCMC', remove_old=False, **kwargs):
        '''
        Loads the model if the model has been previously trained, otherwise trains the model
        Params:
            sessions_id (array of int): gives the sessions to be used for the training/loading (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
            train_method (string): gives the training method when the model is to be trained. today only MCMC is implemented
            remove_old (boolean): removes old saved files
        '''
        if sessions_id is None:
            sessions_id = np.arange(len(self.actions))
        if remove_old:
            self.remove(sessions_id, train_method)
        if train_method!='MCMC':
            raise NotImplementedError
        path = self.build_path(train_method, self.session_uuids[sessions_id])
        if os.path.exists(path):
            self.load(sessions_id, train_method)
            print('results found and loaded')
        else:
            print('training model')
            self.train(sessions_id, train_method, **kwargs)

    def train(self, sessions_id, train_method, **kwargs):
        '''
        Training method
        Params:
            sessions_id (array of int): gives the sessions to be used for the training (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
            train_method (string): the training method. today only MCMC is implemented 
        Output:
            a saved pickle file with the posterior distribution       
        '''
        if train_method=='MCMC':
            std_RW = utils.look_up(kwargs, 'std_RW', 1e-2)
            nb_chains = utils.look_up(kwargs, 'nb_chains', 4)
            nb_steps = utils.look_up(kwargs, 'nb_steps', 1000)
            initial_point = utils.look_up(kwargs, 'initial_point', None)
            print('Launching MCMC procedure with {} chains, {} steps and {} std_RW'.format(nb_chains, nb_steps, std_RW))
            self.params_list, self.lkd_list = self.mcmc(sessions_id, std_RW=std_RW, nb_chains=nb_chains, nb_steps=nb_steps, initial_point=initial_point)
            path = self.build_path(train_method, self.session_uuids[sessions_id])
            pickle.dump([self.params_list, self.lkd_list], open(path, 'wb'))
            print('results of inference SAVED')
        else:
            return NotImplemented

    def load(self, sessions_id, train_method):
        '''
        Load method. This method should not be called directly. Call the load_or_train() method instead
        Params:
            sessions_id (array of int): gives the sessions to be used for the loading (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
            train_method (string): the training method. today only MCMC is implemented 
        Ouput:
            Loads an existing file.         
        '''
        if train_method=='MCMC':
            path = self.build_path(train_method, self.session_uuids[sessions_id])
            [self.params_list, self.lkd_list] = pickle.load(open(path, 'rb'))
        else:
            return NotImplemented

    def remove(self, sessions_id, train_method):
        '''
        Remove method. This method removes the past saved results.
        Params:
            sessions_id (array of int): gives the sessions used for the training (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
            train_method (string): the training method. today only MCMC is implemented 
        Ouput:
            Removes the previously found results        
        '''
        path = self.build_path(train_method, self.session_uuids[sessions_id])
        if os.path.exists(path):
            os.remove(path)
            print('results deleted')
        else:
            print('no results were saved')

    # act=self.actions; stim=self.stimuli; side=self.stim_side
    def compute_prior(self, act, stim, side, sessions_id=None, parameter_type='posterior_mean', train_method='MCMC'):
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
            train_method (string): the training method. today only MCMC is implemented 
        Ouput:
            Computes the prior and accuracy for given act/stim/side   
        '''        
        if not hasattr(self, 'params_list'):
            if sessions_id is None:
                sessions_id = np.arange(len(self.actions))
            path = self.build_path(train_method, self.session_uuids[sessions_id])
            if os.path.exists(path):
                self.load(sessions_id, train_method)
            else:
                print('call the load_or_train() function')

        if train_method=='MCMC': 
            assert(parameter_type in ['MAP', 'posterior_mean', 'whole_posterior']), 'parameter_type must be MAP, posterior_mean or whole_posterior'
        if train_method!='MCMC':
            return NotImplemented
        if train_method=='MCMC' and (parameter_type in ['whole_posterior']):
            return NotImplemented

        if parameter_type=='posterior_mean':
            nb_steps = len(self.params_list)
            parameters_chosen = self.params_list[int(nb_steps/2):].mean(axis=(0,1))[np.newaxis]
        elif parameter_type=='maximum_a_posteriori':
            xmax, ymax = np.where(self.lkd_list==np.max(self.lkd_list))
            parameters_chosen = self.params_list[xmax[0], ymax[0]][np.newaxis]
        if len(act.shape)==1:
            act, stim, side = act[np.newaxis], stim[np.newaxis], side[np.newaxis]
        loglkd, priors = self.evaluate(parameters_chosen, return_details=True, act=act, stim=stim, side=side)
        accuracy = np.exp(loglkd/np.sum(act!=0))

        return np.squeeze(priors), loglkd, accuracy

    def score(self, sessions_id_test, sessions_id, parameter_type='posterior_mean', train_method='MCMC', remove_old=False, param=None):
        '''
        Scores the model on session_id. NB: to implement cross validation, do not train and test on the same sessions
        Params:
            sessions_id (array of int): gives the sessions used for the training (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
            sessions_id_test (array of int): gives the sessions on which we want to test             
            parameter_type (string) : how the prior is computed wrt the parameters. 'posterior_mean' and 'maximum_a_posteriori' are available
            train_method (string): the training method. today only MCMC is implemented
        Outputs:
            accuracy and loglkd on new sessions
        '''
        path = self.build_path(train_method, self.session_uuids[sessions_id], self.session_uuids[sessions_id_test])
        if (remove_old or param is not None) and os.path.exists(path):
            os.remove(path)
        elif os.path.exists(path):
            [prior, loglkd, accuracy] = pickle.load(open(path, 'rb'))
            print('saved score results found')
            print('accuracy on test sessions: {}'.format(accuracy))
            return loglkd, accuracy

        self.load(sessions_id, train_method)
        act, stim, side = self.actions[sessions_id_test], self.stimuli[sessions_id_test], self.stim_side[sessions_id_test]

        if param is not None:
            print('custom parameters: {}'.format(param))
            loglkd, priors = self.evaluate(param[np.newaxis], return_details=True, act=act, stim=stim, side=side)
            accuracy = np.exp(loglkd/np.sum(act!=0))
            print('accuracy on test sessions: {}'.format(accuracy))
            return loglkd, accuracy
        else:
            prior, loglkd, accuracy = self.compute_prior(act, stim, side, sessions_id=sessions_id, parameter_type=parameter_type, train_method=train_method)
            pickle.dump([prior, loglkd, accuracy], open(path, 'wb'))
            print('accuracy on test sessions: {}'.format(accuracy))
        return loglkd, accuracy


    def build_path(self, train_method, l_sessionuuids_train, l_sessionuuids_test=None):
        '''
        Generates the path where the results will be saved
        Params:
            l_sessionuuids_train (array of int)
            train_method (string): the training method. today only MCMC is implemented 
            l_sessionuuids_test (array of int)
        Ouput:
            formatted path where the results are saved
        '''        
        str_sessionuuids = ''
        for k in range(len(l_sessionuuids_train)): str_sessionuuids += '_sess{}_{}'.format(k+1, l_sessionuuids_train[k])
        if l_sessionuuids_test is None:
            path = self.path_results_mouse + 'train_{}_train{}.pkl'.format(train_method, str_sessionuuids)
            return path
        else:
            str_sessionuuids_test = ''
            for k in range(len(l_sessionuuids_test)): str_sessionuuids_test += '_sess{}_{}'.format(k+1, l_sessionuuids_test[k])
            path = self.path_results_mouse + 'train_{}_train{}_test{}.pkl'.format(train_method, str_sessionuuids, str_sessionuuids_test)
            return path

    # act=self.actions; stim=self.stimuli; side=self.stim_side
    def get_parameters(self, sessions_id=None, parameter_type='posterior_mean', train_method='MCMC'):
        '''
        get parameters method.
        Params:
            sessions_id (array of int): gives the sessions on which the training took place (for instance, if you have 4 sessions,
                and you want to train only of the first 3, put sessions_ids = np.array([0, 1, 2]))
            parameter_type (string) : how the prior is computed wrt the parameters. 'posterior_mean', 'maximum_a_posteriori' and 'all' are available
            train_method (string): the training method. today only MCMC is implemented      
        '''        
        if not hasattr(self, 'params_list'):
            if sessions_id is None:
                sessions_id = np.arange(len(self.actions))
            path = self.build_path(train_method, self.session_uuids[sessions_id])
            if os.path.exists(path):
                self.load(sessions_id, train_method)
            else:
                print('call the load_or_train() function')

        if train_method=='MCMC': 
            assert(parameter_type in ['maximum_a_posteriori', 'posterior_mean', 'all']), 'parameter_type must be maximum_a_posteriori, posterior_mean or all'
        if train_method!='MCMC':
            return NotImplemented

        if parameter_type=='posterior_mean':
            nb_steps = len(self.params_list)
            parameters_chosen = self.params_list[int(nb_steps/2):].mean(axis=(0,1))
            return parameters_chosen
        elif parameter_type=='maximum_a_posteriori':
            xmax, ymax = np.where(self.lkd_list==np.max(self.lkd_list))
            parameters_chosen = self.params_list[xmax[0], ymax[0]]
            return parameters_chosen
        else:
            return self.params_list
        

