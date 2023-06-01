import pickle
import numpy as np
from behavior_models import utils as but, models

from one.api import ONE
import brainbox.io.one as bbone
from brainwidemap import bwm_query
import sys

try:
    session_idx = int(sys.argv[1])
except:
    session_idx = 0
    pass

BehaviorModel = models.StimulusKernel
# ONE
one = ONE(base_url="https://openalyx.internationalbrainlab.org")
bwm_df = bwm_query()
sessions = bwm_df.eid.unique()

# load session
eid = sessions[session_idx]
subject = bwm_df[bwm_df.eid == eid].iloc[0].subject
sess_loader = bbone.SessionLoader(one, eid)
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
session_uuids = np.array([eid])
nb_sessions = len(actions)

# instantiate model
model = BehaviorModel(
        path_to_results="results_behavioral/",
        session_uuids=session_uuids,
        mouse_name=subject,
        actions=actions,
        stimuli=stimuli,
        stim_side=stim_side,
        single_zeta=False,
    )

# train
model.load_or_train(remove_old=False, adaptive=True)
