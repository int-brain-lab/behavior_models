# import data
import numpy as np
import utils
from oneibl.one import ONE
one = ONE()
mice_names, ins, ins_id, sess_id, _ = utils.get_bwm_ins_alyx(one)
stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []
for i in range(len(sess_id)):
	if mice_names[i] == mice_names[1]: # take only sessions of first mice
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
from models import expSmoothing_stimside, expSmoothing_prevAction, smoothing_stimside

# get prior
smoothing_s = smoothing_stimside.smoothing_stimside('./results/', session_uuids, mice_names[0], actions, stimuli, stim_side)
smoothing_s.load_or_train(sessions_id=np.array([0,1]), nb_steps=1000, std_RW=0.05, remove_old=False)
loglkd, accuracy = smoothing_s.score(sessions_id_test=np.array([2]), sessions_id=np.array([0, 1]))

expSmoothing_s = expSmoothing_stimside.expSmoothing_stimside('./results/', session_uuids, mice_names[0], actions, stimuli, stim_side)
expSmoothing_s.load_or_train(sessions_id=np.array([0, 1]), nb_steps=1000, std_RW=0.02, remove_old=True)
loglkd, accuracy = expSmoothing_s.score(sessions_id_test=np.array([3]), sessions_id=np.array([0, 1, 2, 4, 5]))
print(accuracy)

expSmoothing_a = expSmoothing_prevAction.expSmoothing_prevAction('./results/', session_uuids, mice_names[0], actions, stimuli, stim_side)
expSmoothing_a.load_or_train(sessions_id=np.array([0, 1]), nb_steps=1000, std_RW=0.02, remove_old=True)
loglkd, accuracy = expSmoothing_a.score(sessions_id_test=np.array([2]), sessions_id=np.array([0, 1]))
print(accuracy)

