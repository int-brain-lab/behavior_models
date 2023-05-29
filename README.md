# behavior models

This is the readme for the behavioral models in the IBL task. 
- the Bayes optimal model
- the exponential smoothing model based on the previous actions
- the exponential smoothing model based on the previous stimulus sides
- the exponential smoothing model based on the previous stimulus sides with asymmetrical learning rates. This last model assumes 4 learning rates: different learning rates are applied when updating the values associated with each side depending on whether the side was chosen (or unchosen) and rewarded (or unrewarded).

See the `main.py` file for an example on prior generation

In the `models` folder, you will find a file called `model.py` from which all models inherits. In this file, you will find all the methods to which you have access. The other files defines the specificities for each model.

The inference takes some minutes but once it has run (and has been saved automatically), model evaluation is very fast.
