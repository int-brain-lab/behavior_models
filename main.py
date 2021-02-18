# load ONE and mice
import numpy as np
import utils
from oneibl.one import ONE
one = ONE()
mice_names, ins, ins_id, sess_id, _ = utils.get_bwm_ins_alyx(one)
stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []

# select particular mice
mouse_name = 'NYU-12'
for i in range(len(sess_id)):
	if mice_names[i] == mouse_name: # take only sessions of first mice
	    data = utils.load_session(sess_id[i])
	    if data['choice'] is not None and data['probabilityLeft'][0]==0.5:
	        stim_side, stimuli, actions, pLeft_oracle = utils.format_data(data)
	        stimuli_arr.append(stimuli)
	        actions_arr.append(actions)
	        stim_sides_arr.append(stim_side)
	        session_uuids.append(sess_id[i])

# format data
stimuli, actions, stim_side = utils.format_input(stimuli_arr, actions_arr, stim_sides_arr)
session_uuids = np.array(session_uuids)

# import models
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prevAction
from models.optimalBayesian import optimal_Bayesian as optBay
from models.biasedApproxBayesian import biased_ApproxBayesian as baisedApproxBay

'''
If you are interested in fitting (and the prior) of the mice behavior
'''
model = exp_stimside('./results/', session_uuids, mouse_name, actions, stimuli, stim_side)
model.load_or_train()
param = model.get_parameters() # if you want the parameters
priors, llk, accuracy = model.compute_prior() # compute prior

'''
if you are interested in pseudo-sessions. NB the model has to previously be trained
'''
model = exp_stimside('./results/', session_uuids, mouse_name, actions=None, stimuli=None, stim_side=None)
priors, llk, accuracy = model.compute_prior(actions, stimuli, stim_side)