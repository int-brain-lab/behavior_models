from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from one.api import ONE
import joblib

from brainwidemap import bwm_query
from behavior_models import models
from iblutil.util import setup_logger

logger = setup_logger('ibl', level='INFO')
# ROOT_PATH = Path("/datadisk/Data/behavior_models")
ROOT_PATH = Path("/mnt/s1/behavior_models")
SUBJECT_EXCLUDES = ['UCLA034', 'UCLA035', 'UCLA036', 'UCLA037', 'PL015', 'PL016', 'PL017', 'PL024', 'NR_0017',
                   'NR_0019', 'NR_0020', 'NR_0021', 'NR_0027']
NCPUS = 12
one = ONE(base_url="https://alyx.internationalbrainlab.org")

df_bwm = bwm_query()
subjects = df_bwm['subject']


def get_alyx_sessions(subject):
    file_sessions = ROOT_PATH.joinpath(subject, '_ibl_subjectSessions.table.pqt')
    if not file_sessions.exists():
        alyx_sessions = one.alyx.rest('sessions', 'list', subject=subject)
        alyx_sessions = pd.DataFrame(alyx_sessions)
        file_sessions.parent.mkdir(exist_ok=True, parents=True)
        alyx_sessions.to_parquet(file_sessions)
    else:
        alyx_sessions = pd.read_parquet(file_sessions)
    return alyx_sessions


def get_trials(subject):
    file_trials = ROOT_PATH.joinpath(subject, '_ibl_subjectTrials.table.pqt')
    if not file_trials.exists():
        file_trials_cache = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table', download_only=True)
        file_trials.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(file_trials_cache, file_trials)
    return pd.read_parquet(file_trials)


def fit_subject(subject):
    model_file = ROOT_PATH.joinpath(subject, 'action_kernel.pqt')
    if model_file.exists():
        return
    if subject in SUBJECT_EXCLUDES:
        return
    trials = get_trials(subject)
    get_alyx_sessions(subject)
    # load and select trials according to session type
    ibiased = trials['task_protocol'].apply(lambda x: x.startswith('_iblrig_tasks_biased') | x.startswith('_iblrig_tasks_ephys'))
    df_trials = trials[ibiased]
    # loop over sessions and fit the action kernek model to each
    nb_sessions = df_trials['session'].unique().size
    logger.info(f'Loading {nb_sessions} sessions for subject {subject}')
    df_sessions = df_trials.groupby('session').agg(
        n_trials=pd.NamedAgg(column="session", aggfunc="size"),
    )
    for f in ['learning_rate', 'sensory_noise', 'lapse_rate_pos', 'lapse_rate_neg']:
        df_sessions[f] = np.NaN
    for eid, _ in df_sessions.iterrows():
        my_model = models.ActionKernel(
            path_to_results=ROOT_PATH,
            session_uuids=[eid],
            mouse_name=subject,
            df_trials=df_trials[df_trials['session'] == eid],
        )
        my_model.load_or_train(remove_old=False, adaptive=True)
        # learning rate, sensory noise, (lapse rate pos, lapse rate neg)
        params = my_model.get_parameters(parameter_type='posterior_mean')
        for i, f in enumerate(['learning_rate', 'sensory_noise', 'lapse_rate_pos', 'lapse_rate_neg']):
            df_sessions.loc[eid, f] = params[i]
    df_sessions.to_parquet(model_file)

# fitting the action kernel if a very low memory task, so we can run in parallel, here we use 12 cores
joblib.Parallel(n_jobs=NCPUS)(joblib.delayed(fit_subject)(subject) for subject in subjects.unique())
