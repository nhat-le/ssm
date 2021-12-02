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

def run_and_save(animal, seed):
    print(f'Starting run and save for {animal}, seed {seed}')
    # Load data
    version = '_113021'
    version_save = '_113021'
    filepath = f'/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/expdata/{animal}_all_sessions{version}.mat'
    fitrangefile = '/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/expdata/fitranges_102121.mat'
    datarange = smart.loadmat(fitrangefile)
    fitrange = datarange['ranges'][datarange['animals'] == animal][0]
    obs, lengths, dirs, fnames, rawchoices = load_multiple_sessions(filepath, fitrange, trialsperblock=15)

    # Find the foraging efficiencies of all blocks
    block_lens = []
    block_corrs = []
    for i in range(len(rawchoices)):
        arr = rawchoices[i]
        block_corrs += list(np.nansum(arr == 1, axis=1))
        block_lens += list(np.sum(~np.isnan(arr), axis=1))

    # Run the fitting procedure
    N_iters = 3000
    obs_dim = obs.shape[1]
    num_states = 4

    np.random.seed(seed)
    masks = ~np.isnan(obs)
    obsmasked = obs[:]
    obsmasked[~masks] = 1

    hmm = ssm.HMM(num_states, obs_dim, observations="blocklapse")
    hmm_lls = hmm.fit(obs, method="em", masks=masks, num_iters=N_iters, init_method="kmeans")

    # Pool states for efficiencies
    zstates = hmm.most_likely_states(obs)
    effs = []
    for i in range(num_states):
        # Find the average foraging efficiency of that state
        blen_state = np.array(block_lens)[zstates == i]
        bcorr_state = np.array(block_corrs)[zstates == i]
        eff_state = sum(bcorr_state) / sum(blen_state)
        effs.append(np.mean(bcorr_state / blen_state))

    # Save the result
    # Save the result
    transmat = hmm.transitions.transition_matrix
    params = hmm.observations.params
    savepath = f'/Users/minhnhatle/Dropbox (MIT)/Sur/MatchingSimulations/expdata/{animal}_hmmblockfit{version_save}.mat'

    vars = ['zstates', 'dirs', 'lengths', 'transmat', 'params',
            'fitrange', 'filepath', 'obs', 'seed', 'hmm_lls', 'effs', 'block_lens', 'block_corrs']
    savedict = make_savedict(vars, locals())

    savefile = 1
    if savefile and not os.path.exists(savepath):
        scipy.io.savemat(savepath, savedict)
        print('File saved')
    elif os.path.exists(savepath):
        print('File exists, skipping save..')


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
    N_iters = 3000
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
    for animal in animals:
        try:
            bestseed = run_animal(animal, seeds)

            # Run and save with the best seed
            run_and_save(animal, bestseed)
        except:
            continue
