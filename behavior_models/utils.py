import gc
import subprocess
from pathlib import Path

import numpy as np
from scipy.special import digamma, betainc, logsumexp
import pandas as pd
import torch
from torch.distributions.normal import Normal


unsqueeze = lambda x: torch.unsqueeze(torch.unsqueeze(x, 0), -1)


def compute_logp_ch_and_pe(values, stim, act, side, zetas, lapses, repetition_bias):
    values = torch.clamp(values, min=1e-7, max=1 - 1e-7)
    pLeft = combine_lkd_prior(stim, zetas, values)
    pRight = 1 - pLeft
    pActions = torch.stack((pRight / (pRight + pLeft), pLeft / (pRight + pLeft)))

    belief = pActions[0] * (torch.unsqueeze(act, 1) == -1) + pActions[1] * (
        torch.unsqueeze(act, 1) == 1
    )
    correct = (act == side) * 1
    prediction_error = torch.unsqueeze(correct, 1) - belief

    unsqueezed_lapses = torch.unsqueeze(lapses, 0)

    # dimension are (left/right x nb_sessions x nb_chains x nb_trials)
    if repetition_bias is not None:
        unsqueezed_rep_bias = torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(repetition_bias, 0), 0), -1
        )
        pActions[:, :, :, 0] = (
            pActions[:, :, :, 0] * (1 - unsqueezed_lapses[:, :, :, 0])
            + unsqueezed_lapses[:, :, :, 0] / 2.0
        )
        pActions[:, :, :, 1:] = (
            pActions[:, :, :, 1:]
            * (1 - unsqueezed_lapses[:, :, :, 1:] - unsqueezed_rep_bias)
            + unsqueezed_lapses[:, :, :, 1:] / 2.0
            + unsqueezed_rep_bias
            * torch.unsqueeze(
                torch.stack(((act[:, :-1] == -1) * 1, (act[:, :-1] == 1) * 1)), 2
            )
        )
    else:
        pActions = pActions * (1 - unsqueezed_lapses) + unsqueezed_lapses / 2.0

    p_ch = (
        pActions[0] * (torch.unsqueeze(act, 1) == -1)
        + pActions[1] * (torch.unsqueeze(act, 1) == 1)
        + 1 * (torch.unsqueeze(act, 1) == 0)
    )
    p_ch = torch.clamp(p_ch, min=1e-7, max=1 - 1e-7)

    logp_ch = torch.log(p_ch)

    return logp_ch, pActions, prediction_error


def combine_lkd_prior(c_t, zetas, pi_t, sigma=0.49):
    sigma_star = torch.sqrt(1 / (sigma**-2 + zetas**-2))
    prior_contrib = Normal(loc=0, scale=1).icdf(1 - pi_t)
    combined = c_t[:, None] / zetas - zetas / sigma_star * prior_contrib
    out = Normal(loc=0, scale=1).cdf(combined)
    return torch.clamp(out, min=1e-7, max=1 - 1e-7)


def get_parameters(lapse_pos, lapse_neg, side, zeta_pos, zeta_neg):
    lapses = (
        unsqueeze(lapse_pos) * (torch.unsqueeze(side, 1) > 0)
        + unsqueeze(lapse_neg) * (torch.unsqueeze(side, 1) < 0)
        + 0.5
        * (unsqueeze(lapse_neg) + unsqueeze(lapse_pos))
        * (torch.unsqueeze(side, 1) == 0)
    )

    zetas = (
        unsqueeze(zeta_pos) * (torch.unsqueeze(side, 1) > 0)
        + unsqueeze(zeta_neg) * (torch.unsqueeze(side, 1) < 0)
        + 0.5
        * (unsqueeze(zeta_pos) + unsqueeze(zeta_neg))
        * (torch.unsqueeze(side, 1) == 0)
    )
    return lapses, zetas


def build_path(
    path_results_mouse, l_sessionuuids_train, l_sessionuuids_test=None, trial_types=None
):
    """
    Generates the path where the results will be saved
    Params:
        l_sessionuuids_train (array of int)
        l_sessionuuids_test (array of int)
    Ouput:
        formatted path where the results are saved
    """
    str_sessionuuids = ""
    for k in range(len(l_sessionuuids_train)):
        str_sessionuuids += "_{}".format(l_sessionuuids_train[k])
    if l_sessionuuids_test is None:
        path = Path(path_results_mouse).joinpath(f"train{format(str_sessionuuids)}.pkl")
        return path
    else:
        assert (
            trial_types is not None
        ), "trial_types can not be None if l_sessionuuids_test is not None"
        str_sessionuuids_test = ""
        for k in range(len(l_sessionuuids_test)):
            str_sessionuuids_test += "_{}".format(l_sessionuuids_test[k])
        path = Path(path_results_mouse).joinpath(f"train{str_sessionuuids}_test{str_sessionuuids_test}_trialtype_{trial_types}.pkl")
        return path


def format_input(stimuli_arr, actions_arr, stim_sides_arr, pLeft_arr=None):
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
    if pLeft_arr is not None:
        pLeft = np.array(
            [
                np.concatenate((pLeft_arr[k], np.zeros(max_len - len(pLeft_arr[k]))))
                for k in range(len(pLeft_arr))
            ]
        )
        return stimuli, actions, stim_side, pLeft
    return stimuli, actions, stim_side


def look_up(dic, key, val):
    if key in dic.keys():
        return dic[key]
    else:
        return val


def trunc_exp(n, tau, lb, ub):
    return np.exp(-n / tau) * (n >= lb) * (n <= ub)


def hazard_f(x, tau, lb, ub):
    return trunc_exp(x, tau, lb, ub) / np.sum(
        trunc_exp(np.linspace(x, x + ub, ub + 1), tau, lb, ub), axis=0
    )


def perform_inference(stim_side, tau=60, gamma=0.8, lb=20, ub=100):
    nb_trials, nb_blocklengths, nb_typeblocks = len(stim_side), ub, 3
    h = np.zeros([nb_trials, nb_blocklengths, nb_typeblocks])
    priors = np.zeros([nb_trials, nb_blocklengths, nb_typeblocks]) - np.inf
    # at the beginning of the task (0), current length is 1 (0) and block type is unbiased (1)
    h[0, 0, 1], priors[0, 0, 1] = 0, 0
    hazard = hazard_f(np.arange(1, ub + 1), tau=tau, lb=lb, ub=ub)
    l = np.concatenate(
        (
            np.expand_dims(hazard, -1),
            np.concatenate(
                (np.diag(1 - hazard[:-1]), np.zeros(len(hazard) - 1)[np.newaxis]),
                axis=0,
            ),
        ),
        axis=-1,
    )
    b = np.zeros([len(hazard), 3, 3])
    b[1:][:, 0, 0], b[1:][:, 1, 1], b[1:][:, 2, 2] = 1, 1, 1  # case when l_t > 0
    b[0][0][-1], b[0][-1][0], b[0][1][np.array([0, 2])] = (
        1,
        1,
        1.0 / 2,
    )  # case when l_t = 1
    # transition matrix l_{t-1}, b_{t-1}, l_t, b_t
    t = np.log(
        np.swapaxes(l[:, :, np.newaxis, np.newaxis] * b[np.newaxis], 1, 2)
    ).reshape(nb_typeblocks * nb_blocklengths, -1)
    priors = priors.reshape(-1, nb_typeblocks * nb_blocklengths)
    h = h.reshape(-1, nb_typeblocks * nb_blocklengths)

    for i_trial in range(nb_trials):
        s = stim_side[i_trial]
        loglks = np.log(
            np.array(
                [
                    gamma * (s == -1) + (1 - gamma) * (s == 1),
                    1.0 / 2,
                    gamma * (s == 1) + (1 - gamma) * (s == -1),
                ]
            )
        )

        # save priors
        if i_trial > 0:
            priors[i_trial] = logsumexp(h[i_trial - 1][:, np.newaxis] + t, axis=(0))
        h[i_trial] = priors[i_trial] + np.tile(loglks, ub)

    priors = priors - np.expand_dims(logsumexp(priors, axis=1), -1)
    h = h - np.expand_dims(logsumexp(h, axis=1), -1)
    priors = priors.reshape(-1, nb_blocklengths, nb_typeblocks)
    h = h.reshape(-1, nb_blocklengths, nb_typeblocks)
    marginal_blocktype = np.exp(priors).sum(axis=1)
    marginal_currentlength = np.exp(priors).sum(axis=2)
    pLeft_inferred = (
        marginal_blocktype[:, 0] * (1 - gamma)
        + marginal_blocktype[:, 1] * 0.5
        + marginal_blocktype[:, 2] * gamma
    )

    return pLeft_inferred, marginal_blocktype, marginal_currentlength, priors, h


def format_data(data, return_dataframe=False, eid=None):
    """
    From a trial table, returns the stimuli, actions, stim_side and pLeft_oracle arrays
    :param data: trials dataframe from Session Loader or ALF loader
    :param return_dataframe: (False), if true returns a dataframe
    :param eid: (None), if labeled and Dataframe output is required, labeled a column with the eid
    :return:
    """
    if "feedbackType" in data.keys():
        stim_side = data["choice"] * (data["feedbackType"] == 1) - data["choice"] * (
            data["feedbackType"] == -1
        )
    else:
        stim_side = (np.isnan(data["contrastLeft"]) == False) * 1 - (
            np.isnan(data["contrastRight"]) == False
        ) * 1
    stimuli = np.nan_to_num(data["contrastLeft"]) - np.nan_to_num(data["contrastRight"])
    if "choice" in data.keys():
        actions = data["choice"]
    else:
        actions = pd.Series(dtype="float64").reindex_like(stim_side)
    pLeft_oracle = data["probabilityLeft"]
    if return_dataframe:
        df_out = pd.DataFrame(dict(stim_side=stim_side, stimuli=stimuli, actions=actions,
                                   pLeft_oracle=pLeft_oracle, eid=data.get('eid', eid)))
        return df_out
    else:
        return (
            np.array(stim_side),
            np.array(stimuli),
            np.array(actions),
            np.array(pLeft_oracle),
        )


def get_bwm_ins_alyx(one):
    import datetime

    """
    Return insertions that match criteria :
    - project code
    - session QC not critical (TODO may need to add probe insertion QC)
    - at least 1 alignment
    - behavior pass
    :return:
    ins: dict containing the full details on insertion as per the alyx rest query
    ins_id: list of insertions eids
    sess_id: list of (unique) sessions eids
    """
    ins = one.alyx.rest(
        "insertions",
        "list",
        django="session__project__name__icontains,ibl_neuropixel_brainwide_01,"
        "session__qc__lt,50,"
        "json__extended_qc__alignment_count__gt,0,"
        "session__extended_qc__behavior,1",
    )
    ins_id = [item["id"] for item in ins]
    sess_id = [item["session_info"]["id"] for item in ins]
    mice_names = np.array([item["session_info"]["subject"] for item in ins])
    sess_id, i = np.unique(sess_id, return_index=True)
    time_stamps = []
    for item in ins:
        s = (
            item["session_info"]["start_time"].split(":")[0]
            + ":"
            + item["session_info"]["start_time"].split(":")[1]
        )
        time_stamps.append(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M").timestamp())
    return (
        mice_names[i],
        np.array(ins)[i],
        np.array(ins_id)[i],
        sess_id,
        np.array(time_stamps)[i],
    )


def load_session(sess_id, one):

    trialstypes = [
        "choice",
        "probabilityLeft",
        "feedbackType",
        "feedback_times",
        "contrastLeft",
        "contrastRight",
        "goCue_times",
        "stimOn_times",
    ]

    tmp = one.load_object(sess_id, "trials")
    # Break container out into a dict with labels
    trialdata = {k: tmp[k] for k in trialstypes}

    return trialdata


def exceedance_proba(alpha, Nsamp=1e6):
    # Compute exceedance probabilities for a Dirichlet distribution
    # FORMAT xp = spm_dirichlet_exceedance(alpha,Nsamp)
    #
    # Input:
    # alpha     - Dirichlet parameters
    # Nsamp     - number of samples used to compute xp [default = 1e6]
    #
    # Output:
    # xp        - exceedance probability
    # __________________________________________________________________________
    #
    # This function computes exceedance probabilities, i.e. for any given model
    # k1, the probability that it is more likely than any other model k2.
    # More formally, for k1=1..Nk and for all k2~=k1, it returns p(x_k1>x_k2)
    # given that p(x)=dirichlet(alpha).
    #
    # Refs:
    # Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ
    # Bayesian Model Selection for Group Studies. NeuroImage (in press)
    # __________________________________________________________________________
    # charles.findling
    # this is a translation from matlab to code. We tried to keep the structure
    # the closest possible to the original script

    Nk = len(alpha)
    xp = np.zeros(Nk)

    for i in range(Nsamp):
        # Sample from univariate gamma densities then normalise
        # (see Dirichlet entry in Wikipedia or Ferguson (1973) Ann. Stat. 1,
        # 209-230)
        # ----------------------------------------------------------------------

        r = np.zeros(Nk)
        for k in range(Nk):
            r[k] = np.random.gamma(alpha[k], 1)

        sr = np.sum(r)
        for k in range(Nk):
            r[k] = r[k] / sr

        # Exceedance probabilities:
        # For any given model k1, compute the probability that it is more
        # likely than any other model k2~=k1
        # ----------------------------------------------------------------------
        xp[np.argmax(r)] += 1

    return xp / sum(xp)


def BMS_dirichlet(*args):
    # Bayesian model selection for group studies
    # FORMAT [alpha, exp_r, xp] = spm_BMS (lme, Nsamp, ecp, alpha0)
    #
    # INPUT:
    # lme      - array of log model evidences
    #              rows: subjects
    #              columns: models (1..Nk)
    # Nsamp    - number of samples used to compute exceedance probabilities
    #            (default: 1e6)
    # do_plot  - 1 to plot p(r|y)
    # sampling - use sampling to compute exact alpha
    # ecp      - 1 to compute exceedance probability
    # alpha0   - [1 x Nk] vector of prior model counts
    #
    # OUTPUT:
    # alpha   - vector of model probabilities
    # exp_r   - expectation of the posterior p(r|y)
    # xp      - exceedance probabilities
    #
    # REFERENCE:
    # Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009)
    # Bayesian Model Selection for Group Studies. NeuroImage 46:1004-1017
    # __________________________________________________________________________
    # charles.findling
    # this is a translation from matlab to code. We tried to keep the structure
    # the closest possible to the original script

    if len(args) < 1:
        return ValueError("Invalid number of arguments")

    max_val = np.finfo("float").max
    lme = args[0]
    Ni, Nk = lme.shape
    c = 1.0
    cc = 1e-3
    log_u = np.zeros([Ni, Nk])
    u = np.zeros([Ni, Nk])
    g = np.zeros([Ni, Nk])
    beta = np.zeros(Nk)
    prev_alpha = np.zeros(Nk)
    exp_r = np.zeros(Nk)
    xp = np.zeros(Nk)

    if len(args) < 2:
        N_samp = int(1e6)
    else:
        N_samp = int(args[1])
    if len(args) < 3:
        ecp = 1
    else:
        ecp = args[2]

    # prior observations
    # --------------------------------------------------------------------
    if len(args) < 4:
        alpha0 = np.ones(Nk)
    else:
        alpha0 = args[3]
    alpha = np.array(alpha0)

    # iterative VB estimation
    # ---------------------------------------------------------------------

    while c > cc:

        # compute posterior belief g(i,k)=q(m_i=k|y_i) that model k generated
        # the data for the i-th subject
        for i in range(Ni):
            for k in range(Nk):
                # integrate out prior probabilities of models (in log space)
                log_u[i, k] = lme[i, k] + digamma(alpha[k]) - digamma(np.sum(alpha))

            # prevent overflow
            log_u[i, :] = log_u[i, :] - np.max(log_u[i, :])

            # prevent numerical problems for badly scaled posteriors
            for k in range(Nk):
                log_u[i, k] = np.sign(log_u[i, k]) * np.minimum(
                    max_val, np.abs(log_u[i, k])
                )

            # exponentiate (to get back to non-log representation)
            u[i, :] = np.exp(log_u[i, :])

            # normalisation: sum across all models for i-th subject
            u_i = np.sum(u[i, :])
            g[i, :] = u[i, :] / u_i

        # expected number of subjects whose data we believe to have been
        # generated by model k
        for k in range(Nk):
            beta[k] = np.sum(g[:, k])

        # update alpha
        prev_alpha[:] = alpha
        for k in range(Nk):
            alpha[k] = alpha0[k] + beta[k]

        # convergence?
        c = np.sum(np.abs(alpha - prev_alpha))

    # Compute expectation of the posterior p(r|y)
    # --------------------------------------------------------------------------
    exp_r[:] = alpha / np.sum(alpha)

    # Compute exceedance probabilities p(r_i>r_j)
    # --------------------------------------------------------------------------

    if ecp:
        if Nk == 2:
            # comparison of 2 models
            xp[0] = betainc(alpha[1], alpha[0], 0.5)
            xp[1] = betainc(alpha[0], alpha[1], 0.5)
        else:
            # comparison of >2 models: use sampling approach
            xp = exceedance_proba(alpha, N_samp)

    return [alpha, exp_r, xp]


def estimate_Beta(theta, weights):
    loc = np.sum(theta * weights)
    scale2 = np.sum(weights * ((theta - loc) ** 2))
    a = ((1 - loc) / scale2 - 1 / loc) * (loc**2)
    b = a * (1 / loc - 1)
    return np.maximum(a, 1), np.maximum(b, 1)


def estimate_Gamma(theta, weights):  # gamma for precision of normal = 1./var
    loc = np.sum(theta * weights)
    scale2 = np.sum(weights * ((theta - loc) ** 2))
    a = (loc**2) / scale2
    b = loc / scale2
    return a, 1 / b


def get_cuda_variable():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(type(obj), obj.size())
        except:
            pass


def get_gpu_memory_map():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
    )

    return float(result)


def make_transparent(plt):
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.xticks([], [])
    plt.yticks([], [])


def clean_up(plt):
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)


if __name__ == "__main__":
    from behavior_models import utils as mut

    c_t = torch.tensor([[-0.1250]])
    zeta_ = torch.tensor([0.0001], dtype=torch.float64)
    pLeft = mut.combine_lkd_prior(
        c_t=c_t, zeta=zeta_, pi_t=torch.tensor([[[1.0 - 1e-16]]]), epsilon_t=0
    )
    pRight = mut.combine_lkd_prior(
        c_t=-c_t,
        zeta=zeta_,
        pi_t=torch.tensor([[[1e-16]]], dtype=torch.float64),
        epsilon_t=0,
    )

    Normal(loc=0, scale=1).icdf(1 - torch.tensor([[[1.0 - 1e-16]]]))


def unsqueeze(x):
    return torch.unsqueeze(torch.unsqueeze(x, 0), -1)
