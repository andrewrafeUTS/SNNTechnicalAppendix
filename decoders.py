import numpy as np
import random

def select_max_spiking_cartpole(x) -> int:
    """ Returns an action based off of the softmax decoding method of a set of spike trains.
    """

    if (sum(x[0]) > sum(x[1])):
        return 0;
    elif (sum(x[0]) < sum(x[1])):
        return 1
    else:
        return -1
        
    
    #probs = summed_spikes/np.sum(summed_spikes)
    #choice = np.random.choice(int(x.size/len(x)), 1, p=probs)
    #return np.argmax(summed_spikes)
    #return 0

def select_max_spiking_lunar_lander(x) -> int:
    # Returns an action based off the output neuron that had the most spiking
    # activity from the given spike train.
    
    x0 = sum(x[0])
    x1 = sum(x[1])
    x2 = sum(x[2])
    x3 = sum(x[3])
    
    if (x0 > x1 and x0 > x2 and x0 > x3):
        return 0
    elif (x1 > x0 and x1 > x2 and x1 > x3):
        return 1
    elif (x2 > x0 and x2 > x1 and x2 > x3):
        return 2
    elif (x3 > x0 and x3 > x1 and x3 > x2):
        return 3
    else:
        return -1

def transform_spiking_probs(x, power) -> int:
    #print(x)
    summed_spikes = np.sum(a=x, axis=0)
    #print(f"Sum{summed_spikes}")
    if (sum(summed_spikes) == 0):
        return np.random.randint(0, int(x.size/len(x)))
    
    probs = summed_spikes/np.sum(summed_spikes)
    transformed_probs = probs ** power
    transformed_probs = transformed_probs/np.sum(transformed_probs)
    choice = np.random.choice(int(x.size/len(x)), 1, p=transformed_probs)
    return choice[0]

def first_to_spike(x, IsCartpole) -> int:
    if IsCartpole:
        if (sum(x[0]) == 0):
            return -1
        elif (x[0][0]==x[0][1]):
            return np.random.randint(low=0, high=2)
        else:
            return np.argmax(x)
    else:
        #print(x)
        if (sum(x[0]) == 0):
            return -1
        else:
            Flatten = np.asarray(x).flatten()
            SpikedIndexes = np.where(Flatten==1)
            return np.random.choice(SpikedIndexes[0])
            
