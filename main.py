# load ONE and mice
from models.biasedBayesian import biased_Bayesian
from models.biasedApproxBayesian import biased_ApproxBayesian as baisedApproxBay
from models.optimalBayesian import optimal_Bayesian as optBay
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prevAction
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
import numpy as np
from models import utils
from oneibl.one import ONE
one = ONE()
mice_names, ins, ins_id, sess_id, _ = utils.get_bwm_ins_alyx(one)
stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []

# select particular mice
mouse_name = 'KS022'
for i in range(len(sess_id)):
    if mice_names[i] == mouse_name:  # take only sessions of first mice
        data = utils.load_session(sess_id[i])
        if data['choice'] is not None and data['probabilityLeft'][0] == 0.5:
            stim_side, stimuli, actions, pLeft_oracle = utils.format_data(data)
            stimuli_arr.append(stimuli)
            actions_arr.append(actions)
            stim_sides_arr.append(stim_side)
            session_uuids.append(sess_id[i])

# format data
stimuli, actions, stim_side = utils.format_input(
    stimuli_arr, actions_arr, stim_sides_arr)
session_uuids = np.array(session_uuids)

# import models
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prevAction
from models.optimalBayesian import optimal_Bayesian as optBay

'''
If you are interested in fitting (and the prior) of the mice behavior
'''

model = exp_prevAction('./results/inference/', session_uuids,
                       mouse_name, actions, stimuli, stim_side)

model.load_or_train(remove_old=False)
param = model.get_parameters()  # if you want the parameters
# compute signals of interest
signals = model.compute_signal(
    signal=['prior', 'prediction_error', 'score'], verbose=False)

'''
if you are interested in pseudo-sessions. NB the model has to previously be trained
It will return an Error if the model has not been trained
'''
model = exp_prevAction('./results/inference/', session_uuids, mouse_name, actions=None, stimuli=None, stim_side=None)
signals = model.compute_signal(signal=['prior', 'prediction_error', 'score'], act=actions, stim=stimuli, side=stim_side)
