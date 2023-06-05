import numpy as np
from behavior_models import models
import pandas as pd
from pathlib import Path

from one.api import ONE
import brainbox.io.one as bbone
from brainwidemap import bwm_query
from iblutil.util import setup_logger

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
# todo repetition_bias should be a parameter
# get the possible combinations for the cross validation
combinations = np.lib.stride_tricks.sliding_window_view(np.tile(np.arange(nb_sessions), 2), nb_sessions)[:nb_sessions, :]
training_sessions = combinations[:, :nb_sessions - 1]
testing_sessions = combinations[:, - 1][:, np.newaxis]


df_trials  = []
eids = df_sessions.eid.unique()
for eid in eids:
    sl = bbone.SessionLoader(one, eid)
    sl.load_trials()
    sl.trials['eid'] = eid
    df_trials.append(sl.trials)
df_trials = pd.concat(df_trials)


my_model = models.ActionKernel(
    path_to_results="results_behavioral/",
    session_uuids=eids,
    mouse_name=SUBJECT,
    df_trials=df_trials,
)
# training each iteration while leaving out one session to score
outlist = []
for idx_perm in range(len(training_sessions)):
    logger.info(f"training session: {training_sessions[idx_perm]}, testing session: {testing_sessions[idx_perm]}")
    my_model.load_or_train(sessions_id=training_sessions[idx_perm], remove_old=False, adaptive=True)
    loglkd, acc = my_model.score(sessions_id_test=testing_sessions[idx_perm], sessions_id=training_sessions[idx_perm],
                                 trial_types="all", remove_old=False)
    outlist.append(
        [
            np.NaN,
            idx_perm,
            training_sessions[idx_perm].tolist(),
            testing_sessions[idx_perm].tolist(),
            SUBJECT,
            nb_sessions,
            my_model.name,
            my_model.repetition_bias,
            loglkd,
            acc,
            my_model.get_parameters(
                parameter_type="posterior_mean"
            ).tolist(),
            eids.tolist(),
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


outdf_expected = pd.read_parquet(Path(models.__file__).parent.joinpath('tests', f'outdf_{SUBJECT}.parquet'))

## %%
for k in outdf.columns:
    print(k, outdf[k], outdf_expected[k])
    print('')




a = my_model.compute_signal(parameter_type='posterior_mean')

params = my_model.get_parameters(parameter_type='posterior_mean')
#learning rate, sensory noise, (lapse rate pos, lapse rate neg)
# decay constant (1 / learning rate)
a['prior'].shape


import torch
import gc
gc.collect()
torch.cuda.empty_cache()