# load ONE and mice
import numpy as np
import utils, torch
import pickle

mice = pickle.load(open('mice.pkl', 'rb'))

# select particular mice
mouse_name = 'KS016'

# format data
stimuli_arr, actions_arr, stim_sides_arr = mice[mouse_name]['stimuli'], mice[mouse_name]['actions'], mice[mouse_name]['stim_side']
stimuli, actions, stim_side = utils.format_input(stimuli_arr, actions_arr, stim_sides_arr)
session_uuids = np.array(mice[mouse_name]['session_uuids'])

# import models
from models.expSmoothing_prevAction_SE import expSmoothing_prevAction_SE as exp_prevAction_SE
from models.expSmoothing_stimside_SE import expSmoothing_stimside_SE as exp_stimSide_SE

'''
If you are interested in fitting (and the prior) of the mice behavior
'''
stimulated = torch.randint(2, size=actions.shape)
model = exp_prevAction_SE('./results/inference/', session_uuids, mouse_name, actions, stimuli, stim_side, stimulated)
model.load_or_train(remove_old=False)
param = model.get_parameters(parameter_type='posterior_mean') # if you want the parameters
signals = model.compute_signal(signal=['prior', 'prediction_error', 'maximum_likelihood'], verbose=False) # compute signals of interest

'''
simulations
'''
act_sim = model.simulate(param, stimuli, stim_side, stimulated, nb_simul=50)
# CAREFUL: like the stim/side/actions, the act-sim is padded to produce a tensor. 
# some simulated actions are thus meaningless and must be discarded , e.g.
# [act_sim[k][stim_side[k]!=0] for k in range(len(act_sim))]
