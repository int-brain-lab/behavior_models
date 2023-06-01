import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import behavior_models.models as models

path_fixtures = Path(models.__file__).parent.joinpath('tests')


class TestInstantiateModels(unittest.TestCase):

    def test_instantiate_all_models(self):
        # single_session = pd.DataFrame(
        #     {'actions': actions[0, :], 'stimuli': stimuli[0, :], 'stim_side': stim_side[0, :]})
        # single_session['eid'] = '4d8c7767-981c-4347-8e5e-5d5fffe38534'
        # single_session['subject'] = 'SWC_038'
        # single_session.to_parquet(
        #     "/home/olivier/Documents/PYTHON/00_IBL/behavior_models/behavior_models/tests/single_session.pqt")
        df_trials = pd.read_parquet(path_fixtures.joinpath("single_session.pqt"))
        kwargs = dict(
            path_to_results="results_behavioral/",
            session_uuids=np.array(['4d8c7767-981c-4347-8e5e-5d5fffe38534']),
            mouse_name='SWC_038',
            actions=df_trials['actions'].values[np.newaxis, :],
            stimuli=df_trials['stimuli'].values[np.newaxis, :],
            stim_side=df_trials['stim_side'].values[np.newaxis, :],
            single_zeta=False,
        )
        action_kernel = models.ActionKernel(**kwargs)
        stimulus_kernel = models.StimulusKernel(**kwargs)
        stimulus_kernel_4a = models.StimulusKernel_4alphas(**kwargs)
        optimal_bayes = models.OptimalBayesian(**kwargs)
