# Behaviour models for the IBL task

This repository contains the code to run the behaviour models for the IBL task with biased blocks (more referecences [here](https://www.internationalbrainlab.org)).

Several models are currently implemented:
- `OptimalBayesian`: the Bayes optimal model
- `ActionKernel`: the exponential smoothing model based on the previous actions
- `StimulusKernel`: the exponential smoothing model based on the previous stimulus sides
- `StimulusKernel_4aphas`: the exponential smoothing model based on the previous stimulus sides with asymmetrical learning rates. This last model assumes 4 learning rates: different learning rates are applied when updating the values associated with each side depending on whether the side was chosen (or unchosen) and rewarded (or unrewarded).


See the `example.py` file for an example on prior generation

In the `models` folder, you will find a file called `model.py` from which all models inherits. In this file, you will find all the methods to which you have access. The other files defines the specificities for each model.

The inference takes some minutes but once it has run (and has been saved automatically), model evaluation is very fast.

## Installation

Clone the repository and install in place:
```shell
git clone https://github.com/int-brain-lab/behavior_models.git
cd behavior_models
pip install -e .
```

## Usage
The simplest is to run a behaviour model on a single session as in the [example_single_session.py](./behavior_models/example_single_session.py) file.
