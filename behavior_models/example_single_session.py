import numpy as np
from behavior_models import models

from one.api import ONE
import brainbox.io.one as bbone

# get trial data from an ephys session on open alyx
one = ONE(base_url="https://openalyx.internationalbrainlab.org")
eid = "35ed605c-1a1a-47b1-86ff-2b56144f55af"
sl = bbone.SessionLoader(one, eid)
sl.load_trials()

# instantiate model
my_model = models.ActionKernel(path_to_results="results_behavioral", session_uuids=eid, df_trials=sess_loader.trials, single_zeta=False)

# train - this will save data in the current directory
my_model.load_or_train(remove_old=False, adaptive=True)

# predict trials and eventually join in the original dataframe
df_prior = my_model.predict_trials()
df_trials = sl.trials.join(df_prior, how='left')
