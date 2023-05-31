import numpy as np
import pandas as pd
from prior_pipelines.decoding.functions.utils import load_metadata
from prior_pipelines.params import CACHE_PATH
from tqdm import tqdm
import pickle
from pathlib import Path
from behavior_models.models.utils import format_data as format_data_mut
import itertools

from behavior_models.models.expSmoothing_prevAction import expSmoothing_prevAction
from behavior_models.models.expSmoothing_stimside import expSmoothing_stimside
from behavior_models.models.optimalBayesian import optimal_Bayesian
from behavior_models.models.expSmoothing_stimside_4alphas import (
    expSmoothing_stimside_4alphas,
)

# model, can_have_repBias, with_repBias
list_of_models = [
    #(expSmoothing_stimside, True, False),
    #(expSmoothing_stimside, True, True),
    #(optimal_Bayesian, True, False),
    #(optimal_Bayesian, True, True),
    (expSmoothing_prevAction, False, False),
    #(expSmoothing_stimside_4alphas, True, False),
    #(expSmoothing_stimside_4alphas, True, True),
]


def get_data_cv(eid_files, modality):
    stimuli_arr, actions_arr, stim_sides_arr, session_uuids = (
        [],
        [],
        [],
        [],
    )
    for (index, row) in eid_files.iterrows():
        regressors = pickle.load(open(row.reg_file, "rb"))
        trials_df = (
            regressors[0] if modality == "widefield" else regressors["trials_df"]
        )
        side, stim, act, pLeft = format_data_mut(trials_df)
        if act is not None and pLeft[0] == 0.5:
            stimuli_arr.append(stim)
            actions_arr.append(act)
            stim_sides_arr.append(side)
            session_uuids.append(index)

    # get maximum number of trials across sessions
    max_len = np.array([len(stimuli_arr[k]) for k in range(len(stimuli_arr))]).max()

    # pad with 0 such that we obtain nd arrays of size nb_sessions x nb_trials and convert to arrays
    stimuli = np.array(
        [
            np.concatenate((stimuli_arr[k], np.zeros(max_len - len(stimuli_arr[k]))))
            for k in range(len(stimuli_arr))
        ]
    )
    actions = np.array(
        [
            np.concatenate((actions_arr[k], np.zeros(max_len - len(actions_arr[k]))))
            for k in range(len(actions_arr))
        ]
    )
    stim_side = np.array(
        [
            np.concatenate(
                (stim_sides_arr[k], np.zeros(max_len - len(stim_sides_arr[k])))
            )
            for k in range(len(stim_sides_arr))
        ]
    )
    return np.array(session_uuids), actions, stimuli, stim_side


import sys

try:
    i_subj = int(sys.argv[1]) - 1
except:
    i_subj = 7
    pass

modality = "ephys"  # 'ephys'
assert modality in ["widefield", "ephys"]


bwmdf, _ = load_metadata(CACHE_PATH.joinpath("*_%s_metadata.pkl" % modality).as_posix())

outlist = []
failed = 0
pvalue_lim = 0.05

subject_name = bwmdf["dataset_filenames"].subject.unique()[i_subj]

regressor_files = bwmdf["dataset_filenames"].set_index("subject").xs(subject_name)
if not isinstance(regressor_files, pd.Series):
    eids_files = regressor_files.groupby(["eid"]).first()
    nb_sessions = eids_files.index.size
    if nb_sessions >= 2:
        print("")
        print(f"subject {i_subj}, {subject_name}")
        print("found {} sessions".format(nb_sessions))
        training_sessions = np.array(
            list(itertools.combinations(np.arange(nb_sessions), nb_sessions - 1))
        )
        testing_sessions = np.stack(
            [
                np.delete(np.arange(nb_sessions), training_sessions[k])
                for k in range(len(training_sessions))
            ]
        )
        session_uuids, actions, stimuli, stim_side = get_data_cv(eids_files, modality)
        if len(training_sessions) > 0:
            print("established {} training sessions".format(len(training_sessions)))
            for (model_beh, can_have_repBias, with_repBias) in list_of_models:
                if can_have_repBias:
                    model = model_beh(
                        "./results_final_allsessions_cv/",
                        session_uuids,
                        subject_name,
                        actions,
                        stimuli,
                        stim_side,
                        single_zeta=True,
                        repetition_bias=with_repBias,
                    )
                else:
                    model = model_beh(
                        "./results_final_v1/",
                        session_uuids,
                        subject_name,
                        actions,
                        stimuli,
                        stim_side,
                        single_zeta=True,
                    )
                print(f"**model {model.name}")
                for idx_perm in range(len(training_sessions)):
                    print(f"training session: {training_sessions[idx_perm]}")
                    print(f"testing session: {testing_sessions[idx_perm]}")

                    model.load_or_train(
                        sessions_id=training_sessions[idx_perm],
                        remove_old=False,
                        adaptive=True,
                    )
                    loglkd, acc = model.score(
                        sessions_id_test=testing_sessions[idx_perm],
                        sessions_id=training_sessions[idx_perm],
                        trial_types="all",
                        remove_old=False,
                    )
                    outlist.append(
                        [
                            i_subj,
                            idx_perm,
                            training_sessions[idx_perm].tolist(),
                            testing_sessions[idx_perm].tolist(),
                            subject_name,
                            nb_sessions,
                            model.name,
                            with_repBias,
                            loglkd,
                            acc,
                            model.get_parameters(
                                parameter_type="posterior_mean"
                            ).tolist(),
                            session_uuids.tolist(),
                        ]
                    )


outdf = pd.DataFrame(outlist,
               columns =["i_subj", 
                         "idx_perm", 
                         "training_sessions", 
                         "testing_sessions", 
                         "subject_name", 
                         "nb_sessions", 
                         "model_name", 
                         "with_repBias", 
                         "loglkd",
                         "acc",
                         "posterior_mean",
                         "session_uuids"])

outdf.to_parquet(f'outdf_{subject_name}.parquet')
