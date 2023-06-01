import pickle
import numpy as np
from behavior_models import utils as but, models
import pandas as pd

from one.api import ONE
import brainbox.io.one as bbone
from brainwidemap import bwm_query
import sys
from iblutil.util import setup_logger
import itertools

logger = setup_logger('ibl', level='INFO')


SUBJECT = 'NYU-45'
model_kwargs = dict(single_zeta=True, repetition_bias=False)

# ONE
one = ONE(base_url="https://openalyx.internationalbrainlab.org")
bwm_df = bwm_query()

df_sessions = bwm_df[bwm_df.subject == SUBJECT]
nb_sessions = df_sessions.eid.unique().size
assert nb_sessions >= 2  # at least 2 sessions


logger.info(f'Loading {nb_sessions} sessions for subject {SUBJECT}')

# get the possible combinations for the cross validation
combinations = np.lib.stride_tricks.sliding_window_view(np.tile(np.arange(nb_sessions), 2), nb_sessions)[:nb_sessions, :]
training_sessions = combinations[:, :nb_sessions - 1]
testing_sessions = combinations[:, - 1][:, np.newaxis]

## %%
df_trials  = []
for eid in df_sessions.eid.unique():
    sl = bbone.SessionLoader(one, eid)
    sl.load_trials()

    side, stim, act, pLeft = but.format_data(sl.trials)
    df_trials_ = pd.DataFrame(dict(side=side, stim=stim, act=act, pLeft=pLeft, eid=eid))
    df_trials.append(df_trials_)

a = pd.concat(df_trials).pivot(columns='eid')


my_model = models.ActionKernel(
    path_to_results="results_behavioral/",
    session_uuids= df_sessions.eid.unique(),
    mouse_name=SUBJECT,
    actions=np.nan_to_num(a['act'].values.T),
    stimuli=np.nan_to_num(a['stim'].values.T),
    stim_side=np.nan_to_num(a['side'].values.T),
)

my_model.load_or_train(training_sessions[0])
