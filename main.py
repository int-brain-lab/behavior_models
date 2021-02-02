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

# get maximum number of trials across sessions
max_len = np.array([len(stimuli_arr[k]) for k in range(len(stimuli_arr))]).max()

# pad with 0 such that we obtain nd arrays of size nb_sessions x nb_trials and convert to arrays
stimuli    = np.array([np.concatenate((stimuli_arr[k], np.zeros(max_len-len(stimuli_arr[k])))) for k in range(len(stimuli_arr))])
actions     = np.array([np.concatenate((actions_arr[k], np.zeros(max_len-len(actions_arr[k])))) for k in range(len(actions_arr))])
stim_side    = np.array([np.concatenate((stim_sides_arr[k], np.zeros(max_len-len(stim_sides_arr[k])))) for k in range(len(stim_sides_arr))])
session_uuids = np.array(session_uuids)

# import models
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prevAction
from models.optimalBayesian import optimal_Bayesian as optBay
from models.biasedBayesian import biased_Bayesian as baisedBay
from models.smoothing_stimside import smoothing_stimside as smooth_stimside

# load and/or run model
model = smooth_stimside('./results/', session_uuids, mouse_name, actions, stimuli, stim_side)
model.load_or_train(nb_steps=2000, remove_old=False) # put 2000 steps for biasedBayesian and smooth_stimside and 1000 for all others
param = model.get_parameters(parameter_type='all') # if you want the parameters
# compute prior (actions,  stimuli and stim_side have been passed as arguments to allow pseudo blocks)
priors, llk, accuracy = model.compute_prior(actions, stimuli, stim_side)
