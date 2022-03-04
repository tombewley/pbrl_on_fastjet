# Preference-based RL on FastJet

## Instructions

`git clone` the following repositories into subfolders of this one:
```
https://github.com/tombewley/rlutils
https://github.com/tombewley/fastjet (NOTE: private)
https://github.com/tombewley/hyperrectangles
```

Commands for running online preference-based RL with various permutations of oracle/human feedback, and a neural network/tree as the reward function model:
```
python run.py oracle tree defaults
python run.py oracle net defaults
python run.py human tree defaults
python run.py human net defaults
```