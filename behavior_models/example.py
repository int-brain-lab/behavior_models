import numpy as np
from behavior_models.models.expSmoothing_prevAction import expSmoothing_prevAction
from behavior_models.models.expSmoothing_stimside import expSmoothing_stimside
from behavior_models.models.optimalBayesian import optimal_Bayesian as CurrentModel
from behavior_models.models import utils as but
import pickle
from one.api import ONE
import brainbox.io.one as bbone
from brainwidemap import bwm_query
import sys

try:
    session_idx = int(sys.argv[1])
except:
    session_idx = 0
    pass

# ONE
one = ONE()
one.alyx.clear_rest_cache()
bwm_df = bwm_query()
sessions = bwm_df.eid.unique()

# load session
session_id = sessions[session_idx]
subject = bwm_df[bwm_df.eid == session_id].iloc[0].subject
sess_loader = bbone.SessionLoader(one, session_id)
if sess_loader.trials.empty:
    sess_loader.load_trials()
trialsdf = sess_loader.trials

# data preprocessing
stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []
data = {
    k: trialsdf[k]
    for k in [
        "choice",
        "probabilityLeft",
        "feedbackType",
        "contrastLeft",
        "contrastRight",
    ]
}
stim_side, stimuli, actions, pLeft_oracle = but.format_data(data)

# format data
stimuli, actions, stim_side = but.format_input(
    [stimuli], [actions], [stim_side]
)
session_uuids = np.array([session_id])
nb_sessions = len(actions)

# instanciate model
model = CurrentModel(
    "results_behavioral/",
    session_uuids,
    subject,
    actions,
    stimuli,
    stim_side,
    single_zeta=False,
)

# train
model.load_or_train(remove_old=False, adaptive=True)
