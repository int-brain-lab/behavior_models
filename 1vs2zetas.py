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
    expSmoothing_prevAction.expSmoothing_prevAction,
    expSmoothing_prevAction.expSmoothing_prevAction,
]

TRAIN = False
# import most recent cached data
bwmdf, _ = load_metadata(CACHE_PATH.joinpath("*_%s_metadata.pkl" % "ephys").as_posix())

uniq_subject = bwmdf["dataset_filenames"].subject.unique()
params_optLkd_1zeta = np.zeros([uniq_subject.size, 4])
params_optLkd_2zetas = np.zeros([uniq_subject.size, 5])
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

            for i_model_type, model_type in enumerate(list_of_models):
                model = model_type(
                    "./results_optLkd_1vs2zetas",
                    session_uuids,
                    subj,
                    actions,
                    stimuli,
                    stim_side,
                    single_zeta=(i_model_type == 1),
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
                        if i_model_type == 1:
                            params_optLkd_1zeta[i_subj] = model.get_parameters(
                                parameter_type="posterior_mean"
                            )
                        else:
                            params_optLkd_2zetas[i_subj] = model.get_parameters(
                                parameter_type="posterior_mean"
                            )

loglkds = loglkds[np.all(loglkds != 0, axis=-1)]
accuracies = accuracies[np.all(accuracies != 0, axis=-1)]
params_optLkd_2zetas = params_optLkd_2zetas[np.all(params_optLkd_2zetas != 0, axis=-1)]
params_optLkd_1zeta = params_optLkd_1zeta[np.all(params_optLkd_1zeta != 0, axis=-1)]

from scipy.stats import pearsonr, wilcoxon

wilcoxon(accuracies[:, 0] - accuracies[:, -1])

df = pd.melt(
    pd.DataFrame(accuracies, columns=["2zetas", "1zeta"]),
    value_vars=["2zetas", "1zeta"],
)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.bar(np.arange(2), accuracies.mean(axis=0), alpha=0.5)
ax = plt.errorbar(
    np.arange(2),
    accuracies.mean(axis=0),
    yerr=accuracies.std(axis=0) / np.sqrt(len(accuracies)),
    fmt="none",
    color="black",
)
plt.ylim(0.65, 0.75)
plt.plot([0, 1], [0.72, 0.72], color="black")
plt.text(0.48, 0.725, "p=0.03")
plt.gca().set_xticks(np.arange(2))
plt.gca().set_xticklabels(["2zetas", "1zeta"])
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().yaxis.set_ticks_position("left")
plt.gca().xaxis.set_ticks_position("bottom")
plt.ylabel("trial-wise likelihood on held-out session")

lr_df = pd.DataFrame(
    np.vstack((params_optLkd_1zeta[:, 0], params_optLkd_2zetas[:, 0])).T,
    columns=["params_optLkd_1zeta", "params_optLkd_2zetas"],
)
plt.subplot(1, 3, 2)
ax = sns.regplot(x="params_optLkd_1zeta", y="params_optLkd_2zetas", data=lr_df)
plt.scatter(params_optLkd_1zeta[:, 0], params_optLkd_2zetas[:, 0], marker="+")
plt.ylim(0, 0.5)
plt.xlim(0, 0.5)
R, p = pearsonr(params_optLkd_1zeta[:, 0], params_optLkd_2zetas[:, 0])
plt.text(0.1, 0.6, "R={}".format(np.round(R, 2)))
plt.text(0.1, 0.5, "p<0.001")
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().yaxis.set_ticks_position("left")
plt.gca().xaxis.set_ticks_position("bottom")
plt.ylabel("learning rate - results_suboptLkd")
plt.xlabel("learning rate - results_optLkd")
plt.suptitle("mouse-level: 1 vs 2 perceptual variabilities")


plt.subplot(1, 3, 3)
dparams = params_optLkd_1zeta[:, 0] - params_optLkd_2zetas[:, 0]
plt.bar(np.arange(1), dparams.mean(axis=0), alpha=0.5)
ax = plt.errorbar(
    np.arange(1),
    dparams.mean(axis=0),
    yerr=dparams.std(axis=0) / np.sqrt(len(dparams)),
    fmt="none",
    color="black",
)
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.gca().yaxis.set_ticks_position("left")
plt.gca().xaxis.set_ticks_position("bottom")
plt.gca().set_ylim(-0.01, 0.01)
plt.gca().set_yticks([-0.008, 0, 0.008])
plt.text(-0.4, 0.008, "p=0.0015")
plt.ylabel("(lr_1zeta) - (lr_2zetas)")

plt.savefig("figures/1vs2zetas_optLik.pdf")
plt.show()
