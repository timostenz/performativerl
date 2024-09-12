# Reproducibility study on Performative Reinforcement Learning

This repository contains code to reproduce and extend the paper [Performative Reinforcement Learning](https://arxiv.org/abs/2207.00046).

The original repository is available here: [https://github.com/gradanovic/icml2023-performative-rl-paper-code](https://github.com/gradanovic/icml2023-performative-rl-paper-code).

### Overview

This repository extends the original repository by the following features:

- feature which estimates the rewards and transition probabilities using samples from the ocupancy measure at each iteration. (no trajectories)
```
python run_experiment.py --sampling --occupancy_iid
```

- additional plots show the state space coverage

- additional plots show the trajectory length

- additional files provide the transition probabilities for the main agent and the follower agents. These files are only generate for the last iteration.

### Structure of the repository

- ```src/``` : This folder contains all the source code files required for generating the experiments' data and figures.
- ```data/``` : This folder is where all the data will be generated.
- ```figures/``` : This folder is where all the figures will be generated.
- ```limiting_envs/``` : This folder is for storing visualizations of the environment.

## Prerequisites:
```
Python3
matplotlib
seaborn
numpy
copy
itertools
time
cvxpy
cvxopt
click
multiprocessing
statistics
json
contextlib
joblib
tqdm
os
cmath
```

## Running the code
To replicate the paper exactly as we did please run the with the following specifications.

### Repeated Policy Optimization (Fig. 2)
```
python run_experiment.py --fbeta=10
```

### Repeated Gradient Ascent (Fig. 3)
```
python run_experiment.py --gradient
```

### Repeated Policy Optimization with Finite Samples (Fig. 4a)
```
python run_experiment.py --sampling
```

### Solving Lagrangian (Fig. 4b)
```
python run_experiment.py --sampling --lagrangian
```

### Sampling from the occupancy measure (Fig. 5)
```
python run_experiment.py --sampling --occupancy_iid
```

## Results

After running the above scripts, new plots will be generated in the figures directory. The output data and the transition probabilities are generated in the data directory.

## Contact Details
For any questions or comments, contact timo.stenz@student.uni-tuebingen.de or jitendra.saripalli@student.uni-tuebingen.de.
