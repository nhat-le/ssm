import autograd.numpy as np
import autograd.numpy.random as npr
import scipy.io
import os
from ssm.exputils import load_multiple_sessions, load_session, make_savedict, run_and_save, get_id_range, sigmoid
npr.seed(0)
from ipywidgets import widgets

import ssm
import smartload.smartload as smart
from ssm.util import find_permutation, split_by_trials
from ssm.plots import gradient_cmap, white_to_color_cmap

import matplotlib.pyplot as plt

# Set the parameters of the HMM
time_bins = 1000   # number of time bins
num_states = 2    # number of discrete states
obs_dim = 30       # dimensionality of observation

# Make an HMM
np.random.seed(128)
true_hmm = ssm.HMM(num_states, obs_dim, observations="blocklapse")

true_hmm.observations.mus = np.array([1, 9]).T
true_hmm.observations.sigmas = np.array([0.8, 1.5]).T
true_hmm.observations.lapses = np.array([0.45, 0.1]).T

# true_hmm.transitions.transition_matrix = np.array([[0.98692759, 0.01307241],
#                                        [0.00685383, 0.99314617]])


# Sample some data from the HMM
true_states, obs = true_hmm.sample(time_bins)
true_ll = true_hmm.log_probability(obs)

arr=  true_hmm.observations.log_likelihoods(obs, None, None, None)

true_ll = true_hmm.log_probability(obs)

data = obs # Treat observations generated above as synthetic data.
N_iters = 200

np.random.seed(123)

## testing the constrained transitions class
hmm = ssm.HMM(num_states, obs_dim, observations="blocklapse")

hmm_lls = hmm.fit(obs, method="em", num_iters=N_iters, initalize=True, init_method="kmeans")