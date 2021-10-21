import scipy.io
import autograd.numpy as np
from ssm.util import split_by_trials



def load_session(filepath, id):
    data = scipy.io.loadmat(filepath)
    choices = data['choices_cell'][0][id][0].astype('float')
    targets = data['targets_cell'][0][id][0].astype('float')

    # flip choices for targets = 0
    signedtargets = 1 - 2 * targets
    signedchoices = (choices * signedtargets + 1) / 2

    bpos = np.where(np.diff(targets))
    bpos = np.hstack([-1, bpos[0], len(targets) - 1])
    blens = np.diff((bpos))
    # blens = np.hstack([bpos[0][0] + 1, blens])
    counters = np.hstack(list(map(lambda x: np.arange(x), blens)))

    choicearr = split_by_trials(signedchoices, blens, chop='max')[:, :15]
    # choicearr[::2,:] = 1 - choicearr[::2,:]
    # choicearr = (choicearr + 1) / 2

    choicearr[np.isnan(choicearr)] = 0
    choicearr[choicearr == 0.5] = 1

    blocktargets = targets[bpos[1:]]

    return choicearr, blocktargets


def load_multiple_sessions(filepath, idlst):
    '''
    filepath: path to extracted data .mat file
    idlst: list of sessions to extract
    returns: concatenated choice, num trials of sessions
    '''
    choicearrs = [load_session(filepath, id)[0] for id in idlst]
    blocktargets = [load_session(filepath, id)[1] for id in idlst]
    blocktargets = np.hstack(blocktargets)
    #     choicearrs = list(map(lambda x: load_session(filepath, x), idlst))

    return np.vstack(choicearrs), [arr.shape[0] for arr in choicearrs], blocktargets


def make_savedict(vars, builtin_names):
    '''
    Make a dictionary of variables to save
    :param vars: a list of variable names
    :return: a dict of var -> value mappings
    '''
    collection = [entry for entry in builtin_names.items() if entry[0] in vars]
    assert len(collection) == len(vars)
    return dict(collection)











