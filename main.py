from braindelphi.params import CACHE_PATH
from braindelphi.decoding.functions.utils import load_metadata
import pickle
from tqdm import tqdm
import numpy as np
from behavior_models.models import utils as but
import itertools
from models import expSmoothing_stimside, expSmoothing_prevAction, optimalBayesian

list_of_models = [expSmoothing_stimside.expSmoothing_stimside,
                  expSmoothing_prevAction.expSmoothing_prevAction]
                  #optimalBayesian.optimalBayesian]

# import most recent cached data
bwmdf, _ = load_metadata(CACHE_PATH.joinpath('*_%s_metadata.pkl' % 'ephys').as_posix())

uniq_subject = bwmdf['dataset_filenames'].subject.unique()
loglkds = np.zeros([uniq_subject.size, len(list_of_models)])
accuracies = np.zeros([uniq_subject.size, len(list_of_models)])
for i_subj, subj in tqdm(enumerate(uniq_subject)):
    subdf = bwmdf['dataset_filenames'][bwmdf['dataset_filenames'].subject == subj]
    stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []
    for index, row in subdf.iterrows():
        out = pickle.load(open(row.reg_file, 'rb'))
        if row.eid not in session_uuids:
            data = {k: out['trials_df'][k] for k in ['choice', 'probabilityLeft', 'feedbackType', 'contrastLeft', 'contrastRight', ]}
            stim_side, stimuli, actions, pLeft_oracle = but.format_data(data)
            stimuli_arr.append(stimuli)
            actions_arr.append(actions)
            stim_sides_arr.append(stim_side)
            session_uuids.append(row.eid)
    # format data
    stimuli, actions, stim_side = but.format_input(stimuli_arr, actions_arr, stim_sides_arr)
    session_uuids = np.array(session_uuids)
    nb_sessions = len(actions)

    if nb_sessions >= 2:
        print('found {} sessions'.format(nb_sessions))
        training_sessions = np.array(list(itertools.combinations(np.arange(nb_sessions), nb_sessions - 1)))
        testing_sessions = np.stack([np.delete(np.arange(nb_sessions), training_sessions[k]) for k in range(len(training_sessions))])
        sel_p = np.array([0.4999]) # [0.33, 0.66, 1.]) #
        p_sess = np.arange(1, len(training_sessions) + 1)/len(training_sessions)
        sel_sess = np.array([np.sum(p_sess <= sel_p[k]) for k in range(len(sel_p))])
        training_sessions = training_sessions[sel_sess]
        testing_sessions  = testing_sessions[sel_sess]

        # import models
        if len(training_sessions) > 0:
            print('established {} training sessions'.format(len(training_sessions)))

            for i_model_type, model_type in enumerate(list_of_models):
                model = model_type('./results/', session_uuids, subj, actions, stimuli, stim_side)
                for idx_perm in range(len(training_sessions)):
                    model.load_or_train(sessions_id=training_sessions[idx_perm], remove_old=False, adaptive=True)
                    loglkds[i_subj, i_model_type], accuracies[i_subj, i_model_type] = model.score(sessions_id_test=testing_sessions[idx_perm],
                                                                                                  sessions_id=training_sessions[idx_perm], remove_old=True)

loglkds = loglkds[np.all(loglkds != 0, axis=-1)]
accuracies = accuracies[np.all(accuracies != 0, axis=-1)]