# import data
import numpy as np
import utils, sys, itertools
from oneibl.one import ONE
one = ONE()
mice_names, ins, ins_id, sess_id, _ = utils.get_bwm_ins_alyx(one)
stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []

try:
    i = int(sys.argv[1]) - 1
except:
    i = 0
    pass

uniq_mice_names = np.unique(mice_names)
mouse_name = uniq_mice_names[i]
print('Case of {} mouse'.format(mouse_name))
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
nb_sessions = len(actions)

print('found {} sessions'.format(nb_sessions))
training_sessions = np.array(list(itertools.combinations(np.arange(nb_sessions), nb_sessions - 1)))
print('established {} training sessions'.format(len(training_sessions)))
testing_sessions = np.stack([np.delete(np.arange(nb_sessions), training_sessions[k]) for k in range(len(training_sessions))])

if len(training_sessions) > 3:
    sel_p = np.array([0.33, 0.66, 1.])
    p_sess = np.arange(1, len(training_sessions) + 1)/len(training_sessions)
    sel_sess = np.array([np.sum(p_sess <= sel_p[k]) - 1 for k in range(len(sel_p))])
    training_sessions = training_sessions[sel_sess]
    testing_sessions  = testing_sessions[sel_sess]

# import models
if len(training_sessions) > 0:
    from models import expSmoothing_stimside, expSmoothing_prevAction, smoothing_stimside, optimalBayesian, biasedBayesian

    list_of_models = [expSmoothing_stimside.expSmoothing_stimside,
                        expSmoothing_prevAction.expSmoothing_prevAction,
                        optimalBayesian.optimal_Bayesian,
                        biasedBayesian.biased_Bayesian]

    for m in list_of_models:
        model = m('./results/', session_uuids, mouse_name, actions, stimuli, stim_side)

        for idx_perm in range(len(training_sessions)):
            if model.name=='biased_bayesian':
                model.load_or_train(sessions_id=training_sessions[idx_perm], nb_steps=2000, remove_old=False)
            else:
                model.load_or_train(sessions_id=training_sessions[idx_perm], nb_steps=100, remove_old=False)           
            loglkd, acc = model.score(sessions_id_test=testing_sessions[idx_perm], sessions_id=training_sessions[idx_perm])




