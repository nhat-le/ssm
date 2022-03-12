import autograd.numpy as np
import autograd.numpy.random as npr
import os
import scipy.io
import numpy as np2
npr.seed(0)

import ssm
import smartload.smartload as smart
from ssm.exputils import load_multiple_sessions, make_savedict
npr.seed(0)

def run_and_validate(animal, seed, params):
    print(f'Starting run and save for {animal}, seed {seed}')
    # Load data
    version = '_113021'
    version_save = '_113021'
    filepath = f'/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/expdata/{animal}_all_sessions{version}.mat'
    fitrangefile = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/expdata/fitranges_102121.mat'
    datarange = smart.loadmat(fitrangefile)
    fitrange = datarange['ranges'][datarange['animals'] == animal][0]
    obs, lengths, dirs, fnames, rawchoices = load_multiple_sessions(filepath, fitrange, trialsperblock=15)


    # Run the fitting procedure
    nstates_lst = params['nstates_lst']
    N_iters = params['N_iters']
    frac_train = params['frac_train']

    # Build the train and test sets
    ntrials, obs_dim = obs.shape
    ntrain = int(ntrials * frac_train)

    np.random.seed(seed)
    masks = ~np.isnan(obs)
    obsmasked = obs[:]
    obsmasked[~masks] = 1

    order = np2.random.permutation(ntrials)
    obs_train = obs[np.sort(order[:ntrain]), :]
    obs_test = obs[np.sort(order[ntrain:]), :]

    # Get the baseline performance (of a Bernoulli model)
    p_bernoulli = np.sum(obs_train) / (np.shape(obs_train)[0] * np.shape(obs_train)[1])
    obs_test_flat = obs_test.flatten()
    ber_LLH = obs_test_flat * np.log(p_bernoulli) + (1 - obs_test_flat) * np.log(1 - p_bernoulli)
    L0 = sum(ber_LLH)


    ll_lst = []
    for num_states in nstates_lst:
        hmm = ssm.HMM(num_states, obs_dim, observations="blocklapse")
        ll_test, obs_train, obs_test = run_hmm_lls(hmm, obs_train, obs_test, masks, N_iters, L0)
        ll_lst.append(ll_test)
        print(f'Num states = {num_states}, likelihood = {ll_test}')

    print(nstates_lst)
    print(ll_lst)

    return ll_lst, nstates_lst, obs_train, obs_test

def run_hmm_lls(hmm, obs_train, obs_test, masks, N_iters, L0):

    hmm.fit(obs_train, method="em", masks=masks, num_iters=N_iters, init_method="kmeans")

    # Evaluate the hmm on the test set
    ll_test = (hmm.log_likelihood(obs_test) - L0) / obs_test.shape[0] / np.log(2)

    return ll_test, obs_train, obs_test




def run_animal(animal, seeds):
    '''
    Given animal name and seed number, run block HMM over all the seeds and report the results
    :param animal: str: animal name
    :param seeds: list[int], seed names
    :return: None
    '''

    # Load data
    version = '_113021'
    filepath = f'/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/expdata/{animal}_all_sessions{version}.mat'
    fitrangefile = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/expdata/fitranges_102121.mat'
    datarange = smart.loadmat(fitrangefile)
    fitrange = datarange['ranges'][datarange['animals'] == animal][0]
    obs, lengths, dirs, fnames, rawchoices = load_multiple_sessions(filepath, fitrange, trialsperblock=15)

    # Run the fitting procedure
    N_iters = 500
    obs_dim = obs.shape[1]
    num_states = 4

    lls_all = []
    for seed in seeds:
        np.random.seed(seed)
        masks = ~np.isnan(obs)
        obsmasked = obs[:]
        obsmasked[~masks] = 1

        hmm = ssm.HMM(num_states, obs_dim, observations="blocklapse")
        hmm_lls = hmm.fit(obs, method="em", masks=masks, num_iters=N_iters, init_method="kmeans")

        lls_all.append(hmm_lls[-1])
        print(f'animal {animal}, seed value = {seed}, hmm LLS = {hmm_lls[-1]:.2f}')

    # Determine the best seed
    idbest = np2.argmax(lls_all)
    print(f'Best seed is: {seeds[idbest]}')
    return seeds[idbest]

if __name__ == '__main__':
    seeds = [121, 122, 123, 124, 125]
    # animals = ['e46']
    # animals = ['f02', 'f03', 'f04', 'f11', 'f12', 'e35', 'e40',
    #     'fh01', 'fh02', 'f05', 'e53', 'fh03', 'f16', 'f17', 'f20', 'f21', 'f22', 'f23']
    # animals = ['e53', 'fh03', 'f16', 'f17', 'f20', 'f21', 'f22', 'f23']
    animals = ['f01']
    params = dict(nstates_lst=np.arange(2, 8),
                  N_iters=500,
                  frac_train=0.8)
    for animal in animals:
        try:
            seed = 123
            # Run and save with the best seed
            run_and_validate(animal, seed, params)
        except:
            continue
