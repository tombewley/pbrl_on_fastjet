# Preference-based RL on FastJet

## Instructions

`git clone` the following repositories into subfolders of this one:
```
https://github.com/tombewley/rlutils
https://github.com/tombewley/fastjet (NOTE: private)
https://github.com/tombewley/hyperrectangles
```

Commands for running online preference-based RL (task = seek out a goal position/attitude) with various permutations of oracle/human feedback, and a neural network/tree as the reward function model:
```
python run.py oracle=0 model.tree_by_variance defaults
python run.py oracle=0 model.net defaults
python run.py human model.tree_by_variance defaults
python run.py human net defaults
```

Alternatively, `run_simplified.py` is a heavily-commented script for running the process on a different task (follow another aircraft on a fixed flight path) with exemplar parameters and reduced use of black box functions from rlutils.