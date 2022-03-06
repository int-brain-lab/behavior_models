# load ONE and mice
import numpy as np
import utils, torch
from one.api import ONE

one = ONE()
mice_names, ins, ins_id, sess_id, _ = utils.get_bwm_ins_alyx(one)
stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []

# select particular mice
mouse_name = 'KS016'
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
stimuli, actions, stim_side = utils.format_input(stimuli_arr, actions_arr, stim_sides_arr)
session_uuids = np.array(session_uuids)

# import models
from models.expSmoothing_prevAction_SE import expSmoothing_prevAction_SE as exp_prevAction_SE
from models.expSmoothing_stimside_SE import expSmoothing_stimside_SE as exp_stimSide_SE
from models.expSmoothing_prevAction_4lr import expSmoothing_prevAction_4lr
from models.expSmoothing_stimSide_4lr import expSmoothing_stimSide_4lr
from models.expSmoothing_prevAction_2lr_2weights import expSmoothing_prevAction_2lr_2weights

'''
If you are interested in fitting (and the prior) of the mice behavior
'''
lr_indexes = torch.randint(2, size=actions.shape)
model = expSmoothing_prevAction_2lr_2weights('./results/inference/', session_uuids, mouse_name, actions, stimuli,
                                             stim_side, lr_indexes)
model.load_or_train(remove_old=False)
param = model.get_parameters()  # if you want the parameters
signals = model.compute_signal(signal=['prior', 'prediction_error', 'maximum_likelihood'],
                               verbose=False)  # compute signals of interest

'''
if you are interested in pseudo-sessions. NB the model has to previously be trained
It will return an Error if the model has not been trained
'''
model = expSmoothing_prevAction_4lr('./results/inference/', session_uuids, mouse_name, actions=None, stimuli=None,
                                    stim_side=None, lr_indexes=lr_indexes)
signals = model.compute_signal(signal=['prior', 'prediction_error', 'maximum_likelihood'], act=actions, stim=stimuli,
                               side=stim_side)
