from code.params import CACHE_PATH
from code.decoding.functions.utils import load_metadata
import pickle
from tqdm import tqdm
import numpy as np
from behavior_models.models import utils as but
import itertools
from models import (
    expSmoothing_stimside,
    expSmoothing_prevAction,
    optimalBayesian,
    utils,
)
import os
import pandas as pd

list_of_models = [
    (expSmoothing_prevAction.expSmoothing_prevAction, "./results_optLkd"),
    (expSmoothing_stimside.expSmoothing_stimside, "./results_optLkd"),
    (optimalBayesian.optimal_Bayesian, "./results_optLkd_unbiased"),
]

TRAIN = False
# import most recent cached data
bwmdf, _ = load_metadata(CACHE_PATH.joinpath("*_%s_metadata.pkl" % "ephys").as_posix())

uniq_subject = bwmdf["dataset_filenames"].subject.unique()
loglkds = np.zeros([uniq_subject.size, len(list_of_models)])
accuracies = np.zeros([uniq_subject.size, len(list_of_models)])
# parameters = {model.name: [] for model in list_of_models}
for i_subj, subj in tqdm(enumerate(uniq_subject)):
    subdf = bwmdf["dataset_filenames"][bwmdf["dataset_filenames"].subject == subj]
    stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []
    for index, row in subdf.iterrows():
        out = pickle.load(
            open(
                CACHE_PATH.as_posix() + row.reg_file.as_posix().split("cache")[-1], "rb"
            )
        )
        if row.eid not in session_uuids:
            data = {
                k: out["trials_df"][k]
                for k in [
                    "choice",
                    "probabilityLeft",
                    "feedbackType",
                    "contrastLeft",
                    "contrastRight",
                ]
            }
            stim_side, stimuli, actions, pLeft_oracle = but.format_data(data)
            stimuli_arr.append(stimuli)
            actions_arr.append(actions)
            stim_sides_arr.append(stim_side)
            session_uuids.append(row.eid)
    # format data
    stimuli, actions, stim_side = but.format_input(
        stimuli_arr, actions_arr, stim_sides_arr
    )
    session_uuids = np.array(session_uuids)
    nb_sessions = len(actions)

    if nb_sessions >= 2:
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
        sel_p = np.array([0.4999])  # [0.33, 0.66, 1.]) #
        p_sess = np.arange(1, len(training_sessions) + 1) / len(training_sessions)
        sel_sess = np.array([np.sum(p_sess <= sel_p[k]) for k in range(len(sel_p))])
        training_sessions = training_sessions[sel_sess]
        testing_sessions = testing_sessions[sel_sess]

        # import models
        if len(training_sessions) > 0:
            print("established {} training sessions".format(len(training_sessions)))

            for i_model_type, (model_type, path_model_type) in enumerate(
                list_of_models
            ):
                model = model_type(
                    path_model_type,
                    session_uuids,
                    subj,
                    actions,
                    stimuli,
                    stim_side,
                    single_zeta=True,
                )
                for idx_perm in range(len(training_sessions)):
                    loadpath = utils.build_path(
                        model.path_results_mouse,
                        model.session_uuids[training_sessions[idx_perm]],
                    )
                    if os.path.exists(loadpath) or TRAIN:
                        model.load_or_train(
                            sessions_id=training_sessions[idx_perm],
                            remove_old=False,
                            adaptive=True,
                        )
                        (
                            loglkds[i_subj, i_model_type],
                            accuracies[i_subj, i_model_type],
                        ) = model.score(
                            sessions_id_test=testing_sessions[idx_perm],
                            sessions_id=training_sessions[idx_perm],
                            remove_old=False,
                        )

loglkds = loglkds[np.all(loglkds != 0, axis=-1)]
accuracies = accuracies[np.all(accuracies != 0, axis=-1)]

from scipy.stats import pearsonr, wilcoxon

wilcoxon(accuracies[:, 0] - accuracies[:, -1])

df = pd.melt(
    pd.DataFrame(accuracies, columns=["2zetas", "1zeta"]),
    value_vars=["2zetas", "1zeta"],
)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.bar(np.arange(3), accuracies.mean(axis=0), alpha=0.5)
ax = plt.errorbar(
    np.arange(3),
    accuracies.mean(axis=0),
    yerr=accuracies.std(axis=0) / np.sqrt(len(accuracies)),
    fmt="none",
    color="black",
)
plt.ylim(0.65, 0.75)
plt.plot([0, 1], [0.72, 0.72], color="black")
plt.text(0.48, 0.723, "p<0.001")
plt.plot([0, 2], [0.73, 0.73], color="black")
plt.text(0.98, 0.733, "p<0.001")
plt.plot([1, 2], [0.7, 0.7], color="black")
plt.text(1.48, 0.703, "p>0.02")
plt.gca().set_xticks(np.arange(3))
plt.gca().set_xticklabels(["actKer", "stimKer", "optBay"])
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().yaxis.set_ticks_position("left")
plt.gca().xaxis.set_ticks_position("bottom")
plt.ylabel("trial-wise likelihood on held-out session")

plt.savefig("figures/optLik_comparison.pdf")
plt.show()
