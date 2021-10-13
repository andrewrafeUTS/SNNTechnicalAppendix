import numpy as np

def normalized_probabilities(x):
    # x expected to be between 0 and 1 anything over 1 will be treated as guaranteed probability of firing
    rand = np.random.rand(x.size)
    x = x/x.sum()
    encoded = np.where(x >= rand, 1, 0)
    return encoded

def probabilities(x):
    rand = np.random.rand(x.size)
    encoded = np.where(x >= rand, 1, 0)
    return encoded

def rate_based_encoding(x, t):
    rand = np.random.rand(x.size)
    x = x/x.sum()
    #print(x)
    #np.clip(x, 0., 1.)
    probs = np.where(x >= rand, 1, 0)
    inverted_probs = 1 - probs
    inverted_probs = inverted_probs * 100 + 1
    inverted_probs = inverted_probs.astype(int)
    #print(inverted_probs)
    spikes = np.where(t%inverted_probs == 0, 1, 0)
    return spikes