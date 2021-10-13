import numpy as np
import sys
import gym
import math

from nodes import InputNode, IzhikevichNode, InputEncoder
from connections import AbstractConnection, EncoderConnection
from network import Network
from monitors import SpikeMonitor
from encoders import probabilities, normalized_probabilities, rate_based_encoding
from decoders import transform_spiking_probs, select_max_spiking_cartpole
import matplotlib.pyplot as plt
from decodingMethod import DecodingMethod

np.seterr(all='raise')
np.set_printoptions(suppress=True)

if len(sys.argv) == 5:
    exposure_period = int(sys.argv[1])
    #lr = 1.3145 * math.exp(-0.015*float(sys.argv[1]))
    time = exposure_period
    #print("eta: " + sys.argv[1] + ", lr: " + str(lr))
    lr = float(sys.argv[2])
    if (str(sys.argv[3]) == "f2f"):
        decoding_method = DecodingMethod.F2F
    elif (str(sys.argv[3]) == "f2freset"):
        decoding_method = DecodingMethod.F2F_RESET
    elif (str(sys.argv[3]) == "rate"):
        decoding_method = DecodingMethod.RATE
    else:
        decoding_method = DecodingMethod.RATE_RESET
    
    decay_strength = float(sys.argv[4])
        
else:
    time = 60
    exposure_period = time
    lr = 0.05
    decay_strength = 0.05
    decoding_method = DecodingMethod.F2F_RESET
    
print("EP: " + str(exposure_period) + ", lr: " + str(lr) + ", decoding: " + str(decoding_method) + ", decay strength: " + str(decay_strength))
#if len(sys.argv) == 3:
#    print("lr=" + sys.argv[1] + ", eta=" + sys.argv[2])
#    lr = float(sys.argv[1])
#    exposure_period = int(sys.argv[2])
#    time = int(sys.argv[2])
#else:
#    time = 1
#    exposure_period = time
#    lr = 1.13145

steps = 200
#alpha = 10

#lr eta = 1 alpha = 0.1: 0.1
#lr eta = 10 alpha = 0.1: 0.01
#lr eta = 25 alpha = 0.1: 0.004
#lr eta = 50 alpha = 5: 0.002
#lr eta = 100 alpha = 5: 0.001

env = gym.make('CartPole-v1')

# Create the layers

inpt = IzhikevichNode(8)
out = IzhikevichNode(2)

#hidden = IzhikevichNode(n=50)

# Create the connections between the layers
in_out = AbstractConnection(source=inpt, target=out, wmin=-20.0, wmax=80.0)
#in_out.w = np.asarray([[59.39248354, -16.74360197, 16.13976146, 38.71771547, -19.44616103,18.7903069, 4.15672829,  37.46479719],[ 28.46200822,  37.10049465,  -4.07208196,   1.07577542, 140.37162112, 10.43282319,  60.18002535,   5.3759153 ]])

#in_hidden = AbstractConnection(source=inpt, target=hidden, wmin=0, wmax=50)
#hidden_out = AbstractConnection(source=hidden, target=out, wmin=0, wmax=50)

in_monitor = SpikeMonitor(inpt)
out_monitor = SpikeMonitor(out)

#hidden_monitor = SpikeMonitor(source=hidden)

# Create the network
net = Network(inpt, out)
net.AddLayer(inpt, "Input Layer")
net.AddLayer(out, "Output Layer")
net.AddConnection(in_out)
net.AddMonitor(in_monitor, "InputSpikeMonitor")
net.AddMonitor(out_monitor, "OutputSpikeMonitor")

#net.add_connection(connection=in_hidden)
#net.add_connection(connection=hidden_out)
#net.add_monitor(monitor=hidden_monitor, name="HiddenSpikeMonitor")

#in_out.presynaptic_normalization(norm=20)
#in_out.postsynaptic_normalization(norm=20)
#in_out.postsynaptic_normalization(norm=20)


def calculate_learning_rate(alpha, exposure_period, total_steps):
    #return alpha / (exposure_period * total_steps)
    #return alpha/exposure_period
    #return alpha
    return lr

def preprocess(obs):
    newObs = np.asarray([
        obs[0] * 10.0  if obs[0] >= 0 else 0,
        abs(obs[0]) * 10.0 if obs[0] < 0 else 0,
        obs[1] * 10.0 if obs[1] >= 0 else 0,
        abs(obs[1]) * 10.0 if obs[1] < 0 else 0,
        obs[2] * 10.0 if obs[2] >= 0 else 0,
        abs(obs[2]) * 10.0 if obs[2] < 0 else 0,
        obs[3] * 10.0 if obs[3] >= 0 else 0,
        abs(obs[3]) * 10.0 if obs[3] < 0 else 0
    ], dtype=np.float32)
    #print(newObs)
    return newObs

def update_weights(network, time, all_actions, all_rewards, all_source_spikes, all_target_spikes, all_total_rewards, all_ltp):
    # Create the trace arrays from the spike trains
    """decay = 0.95
    
    source_trace = np.zeros(shape=(len(source_spikes), source_spikes[0].size ))
    target_trace = np.zeros(shape=(len(target_spikes), target_spikes[0].size))
    
    #print(len(source_spikes))
    #print(source_spikes)

    for i in range(len(source_spikes)):
        #print(source_trace)
        if i == 0:
            source_trace[i,:] = np.where(source_spikes[i] == 1, 1, 0)
            target_trace[i,:] = np.where(target_trace[i] == 1, 1, 0)
        else:
            source_trace[i,:] = np.where(source_spikes[i] == 1, 1, source_trace[i-1,:]*decay)
            target_trace[i,:] = np.where(target_spikes[i] == 1, 1, target_trace[i-1,:]*decay)
    """
    
    #print(len(source_trace))
    #print(source_trace.size)
    # Look for when the presynaptic neurons fired (i.e. when the source trace is 1),
    # then calculate the average trace for the action neuron over the next n timesteps
    # then update that weight by the reward * avg_trace
    # The average trace gives a good indication of post synaptic activity, so if it is not
    # spiking regularly it will have a low avg trace and the more it is spiking the higher
    # the weight change will be.

    # Get the best 20 episodes.
    best_episodes = np.argpartition(np.asarray(all_total_rewards), -20)[-20:]
    #print(len(best_episodes))
    worst_episodes = np.argpartition(np.asarray(all_total_rewards), 20)[:20]
    #print(all_ltp)
    #print(all_ltp)
    #print(len(all_ltp[0]))
    for e in best_episodes:
        #ep_length = int(len(all_source_spikes[e])/time)
        #learning_rate = calculate_learning_rate(alpha, exposure_period, ep_length)
        #learning_rate = lr
        
        for t in range(len(all_ltp[e])):
            #print(str(e) + ", " + str(t) + ", " + str(all_ltp[e][t]))
            #print(all_ltp[e][t])
            #print("HERE")
            #print(all_ltp[e][t])
            network.Connections[0].w += np.where(all_ltp[e][t] != 0, all_ltp[e][t] * lr, 0)
            #print(network.Connections[0].w)
            
    
    for e in worst_episodes:
        #ep_length = int(len(all_source_spikes[e])/time)
        #learning_rate = calculate_learning_rate(alpha, exposure_period, ep_length)
        #learning_rate = lr
        for t in range(len(all_ltp[e])):
            #print(str(e) + ", " + str(t) + ", " + str(all_ltp[e][t]))
            #print(all_ltp[e][t])
            network.Connections[0].w -= np.where(all_ltp[e][t] != 0, all_ltp[e][t] * lr, 0)
            #print(network.Connections[0].w)
            #print(net.Connections[0].w)
            #s = int(t/time)'''
    
    return network

    '''for i in range(all_source_spikes[e][0].size):
                # Figure out what step we are up to
                #print(t)
                #print(sum(reward[s:s+10])-5)
                if all_source_spikes[e][t][i] == 1:
                    forward = 0
                    back = 1
                    while True:
                        #print(forward)
                        #print(len(target_spikes))
                        if (t+forward >= len(all_target_spikes[e]) or forward >= 10):
                            break
                        target_ltp = np.where(all_target_spikes[e][t+forward] == 1, 1, 0)
                        #print(target_ltp)
                        if (np.sum(target_ltp) != 0):
                            in_out.w[:,i] += np.where(target_ltp == 1, np.random.rand() * learning_rate, 0)
                            #print("LTP")
                            break
                        else:
                            if (t-back < 0 or back >= 10):
                                break
                            target_ltd = np.where(all_target_spikes[e][t-back] == 1, 1, 0)
                            if (np.sum(target_ltd) != 0):
                                in_out.w[:,i] -= np.where(target_ltd == 1, np.random.rand() * learning_rate, 0)
                                #print("LTD")
                                break
                        forward += 1
                        back += 1
                        
    for e in worst_episodes:
        ep_length = int(len(all_source_spikes[e])/time)
        #learning_rate = calculate_learning_rate(alpha, exposure_period, ep_length)
        learning_rate = lr
        for t in range(len(all_source_spikes[e])):
            s = int(t/time)
            
            for i in range(all_source_spikes[e][0].size):
                # Figure out what step we are up to
                #print(t)
                #print(sum(reward[s:s+10])-5)
                if all_source_spikes[e][t][i] == 1:
                    forward = 0
                    back = 1
                    while True:
                        #print(forward)
                        #print(len(target_spikes))
                        if (t+forward >= len(all_target_spikes[e]) or forward >= 10):
                            break
                        target_ltp = np.where(all_target_spikes[e][t+forward] == 1, 1, 0)
                        #print(target_ltp)
                        if (np.sum(target_ltp) != 0):
                            in_out.w[:,i] -= np.where(target_ltp == 1, np.random.rand() * learning_rate, 0)
                            #print("LTP")
                            break
                        else:
                            if (t-back < 0 or back >= 10):
                                break
                            target_ltd = np.where(all_target_spikes[e][t-back] == 1, 1, 0)
                            if (np.sum(target_ltd) != 0):
                                in_out.w[:,i] += np.where(target_ltd == 1, np.random.rand() * learning_rate, 0)
                                #print("LTD")
                                break
                        forward += 1
                        back += 1
                """
                if source_trace[t,i] == 1:
                    #print(target_trace[t:t+12, action])
                    #avg_trace = np.average(target_trace[t:t+12, action[int(t/time)]])
                    #print(f"TARGET TRACE:{target_trace[t:t+12,:]}")
                    avg_trace = target_trace[t:t+12,action[int(t/time)]].mean(0)
                    #avg_trace = target_trace[t:t+12,:].mean(0)
                    #avg_trace = avg_trace*2 - 1 # Low activity induces LTD, high activity induces LTP
                    #print(avg_trace)
                    #print(f"AVG TRACE:{avg_trace}")
                    #print(reward[int(t/time)][0])
                    in_out.w[action[int(t/time)],i] += avg_trace * sum(reward)/200 * 0.0025
                    #in_out.w[:,i] += avg_trace * sum(reward)/200 * 0.0025
                    #print(f"CHANGE: {avg_trace * sum(reward) * 0.00025 + 0.1 * (np.random.rand(2)*2-1)}")
                    #print(in_out.w)
                    #in_out.w -= 0.00001
                """
        '''
   
def update_weights_synaptic_trace(time, all_actions, all_rewards, all_source_spikes, all_target_spikes, all_total_rewards):
    synaptic_plasticity = []
    synaptic_plasticity.append(np.zeros(len(net.Monitors["Input Layer"].Values[0])))
    synaptic_plasticity.append(np.zeros(len(net.Monitors["Output Layer"].Values[0])))
    synaptic_plasticity = np.asarray(synaptic_plasticity)
    print(synaptic_plasticity)
    
    #for _ in range(net)

def random_mutations():
    # This function will be called if the rate of improvement stagnates for more than 3 cycles of 100 episodes.
    # Go through each of the weights and change them depending on the chances in the arguments
    for i in range(2): # Output neurons
        for j in range(8): #input neurons
            # Generate a random float
            mutation_chance = np.random.rand()
            if (mutation_chance <= 0.21 and mutation_chance > 0.11):
                in_out.w[i,j] += np.random.rand() * 20
            elif (mutation_chance <= 0.11 and mutation_chance > 0.01):
                in_out.w[i,j] -= np.random.rand() * 20
            elif (mutation_chance <= 0.01):
                in_out.w[i,j] *= -1.0

total_reward_list = []
num_eps = 10001
#in_out.w.fill(10.)
#in_out.presynaptic_normalization(norm=50, source_size=8, target_size=2)
#in_out.postsynaptic_normalization(norm=100)
avg_reward_list = []
#print(in_out.w)
power = 1
steps_to_next_rand_mutation = 5
previous_weights_revert = in_out.w
all_action_list = []
all_reward_list = []
all_in_spike_trains = []
all_out_spike_trains = []
all_total_reward = []
all_ltp = []
reward_steps = []
for e in range(num_eps):
    net.ResetStateVariables(True)
    #print(net.LTP)
    obs = env.reset()
    #sys.stdout.write(f"\r{e/num_eps*100:.1f}%")
    #sys.stdout.flush()
    #print(in_out.w)
    action_list = []
    reward_list = []
    done = False
    step = 0
    while not done and step < steps:
        #env.render()
        inpt = preprocess(obs)
        #enc = probabilities(inpt)
        action, out_time = net.Run(inpt, time, True, decoding_method)
        #print(net.Layers["Output Layer"].Spikes)
        out_strain = out_monitor.Values[int(-1*out_time):]
        #action = select_max_spiking(np.asarray(out_strain))
        action_list.append(action)
        obs, reward, done, info = env.step(action)
        #reward = np.random.rand(1) * max_reward[action]
        reward_list.append(reward)
        step+=1
        #print(reward)
        #print(action)
    #print(reward_list)
    total_reward_list.append(sum(reward_list))
    
    #in_out.presynaptic_normalization(norm=50, source_size=8, target_size=2)
    
    #print(f"\n{in_out.w}") 
    if (e < 100):
        all_action_list.append(action_list)
        all_reward_list.append(reward_list)
        all_in_spike_trains.append(in_monitor.Values)
        all_out_spike_trains.append(out_monitor.Values)
        all_total_reward.append(sum(reward_list))
        all_ltp.append(net.LTP)
        #print("-----------------------" + str(net.LTP))
    else:
        all_action_list[e%100] = action_list
        all_reward_list[e%100] = reward_list
        all_in_spike_trains[e%100] = in_monitor.Values
        all_out_spike_trains[e%100] = out_monitor.Values
        all_total_reward[e%100] = sum(reward_list)
        all_ltp[e%100] = net.LTP
    if e % 100 == 0 and e != 0:
        
        #print(all_in_spike_trains)
        net = update_weights(net, time, all_action_list, all_reward_list, all_in_spike_trains, all_out_spike_trains, all_total_reward, all_ltp)
        #in_out.presynaptic_normalization(norm=50)
        #in_out.decay(decay_strength)
        #in_out.postynaptic_normalization(200)
        #in_out.presynaptic_normalization(100)
        #print(in_out.w)
        print(f"{sum(total_reward_list[-100:])/100}")
        #print(net.Connections[0].w)
        reward_steps.append(sum(total_reward_list[-100:])/100)
        #RANDOM MUTATION SECTION
        if len(reward_steps) >= 5:
            #print("Var" + str(np.var(np.asarray(reward_steps[-5:]))))
            if np.var(np.asarray(reward_steps[-5:])) <= 4.0:
                print("Random Mutation")
                random_mutations()
                
        print(net.Connections[0].w)
        # Do the random mutations if the average reward of the last 100 episodes is less than the average of the last three
        avg_reward = sum(total_reward_list[-100:])/100
        """
        if (steps_to_next_rand_mutation < 0 and len(avg_reward_list) > 5 and avg_reward < sum(avg_reward_list[-5:])/5):
            steps_to_next_rand_mutation = 5
            print("RANDOM MUTATION")
            previous_weights_revert = in_out.w
            random_mutations()
            #in_out.postsynaptic_normalization(norm=150)
            print(in_out.w)

        steps_to_next_rand_mutation -= 1
        """
        avg_reward_list.append(avg_reward)
        if avg_reward >= 195.0:
            break
        
        #print(f"\n{in_out.w}")  

        #plt.plot(avg_reward_list)
        #plt.plot(np.arange(100,num_eps+1,100), avg_reward_list)
        #plt.show()
    
        #in_monitor.Plot("Input Spike Train")
        #out_monitor.Plot("Output Spike Train")

#print(in_out.w)
#print(f"{sum(total_reward_list[-100:])/100}")
#avg_reward_list.append(sum(total_reward_list[-100:])/100)

#print(f"\n{in_out.w}")  

#plt.plot(total_reward_list)
#plt.plot(np.arange(100,num_eps+1,100), avg_reward_list)
#plt.show()

#in_monitor.Plot("Input Spike Train")
#out_monitor.Plot("Output Spike Train")

print(net.Connections[0].w)