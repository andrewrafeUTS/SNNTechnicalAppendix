import numpy as np
import sys
import gym
import math

from nodes import IzhikevichNode, InputEncoder, InputNode
from connections import AbstractConnection, EncoderConnection
from network import Network
from monitors import SpikeMonitor
from encoders import probabilities, normalized_probabilities, rate_based_encoding
from decodingMethod import DecodingMethod
import matplotlib.pyplot as plt

class CartPoleRSTDPDifferentRewards:
    
    def __init__(self, NeuronsPerLayer, ExposurePeriod: int = 50, Decoding: DecodingMethod = DecodingMethod.F2F_RESET):
        # Create the network according to the given parameters
        
        self.Environment = gym.make('CartPole-v1')
        self.Environment.reset()
        
        # Network Values
        #self.LearningRate = 1.0 # TODO: Need to recover values from hyperparameter experiment
        
        self.Decoding = Decoding
        self.ExposurePeriod = ExposurePeriod
        
        # Create the network
        self.Network = self.CreateNetwork(NeuronsPerLayer)
        
        self.LearningRate = 0.06
        
    def CreateNetwork(self, NeuronsPerLayer):
        
        AllLayers = []
        
        # 1: Create the layers
        for i in range(len(NeuronsPerLayer)):
            if i == 0:
                LayerInput = InputNode(NeuronsPerLayer[0])
                AllLayers.append(LayerInput)
            elif i == len(NeuronsPerLayer) - 1:
                LayerOutput = IzhikevichNode(NeuronsPerLayer[i])
                AllLayers.append(LayerOutput)
            else:
                AllLayers.append(IzhikevichNode(NeuronsPerLayer[i]))
                
        # 2: Add the connections between the layers
        
        AllConnections = []
        
        for i in range(len(AllLayers) - 1):
            if i >= 0:
                AllConnections.append(AbstractConnection(
                    source=AllLayers[i],
                    target=AllLayers[i+1],
                    wmin=40.0,
                    wmax=60.0))
                #AllConnections[0].w.fill(18.75)
            else:
                 AllConnections.append(EncoderConnection(
                    source=AllLayers[i],
                    target=AllLayers[i+1],
                    wmin=40.0,
                    wmax=150.0))
                 AllConnections[0].w.fill(50)
            #AllConnections[i].presynaptic_normalization(20)
            
        
        # 3: Create the network by giving the input and output layers
        SNN = Network(LayerInput, LayerOutput)
        for i in range(len(AllLayers)):
            if i == 0:
                SNN.AddLayer(LayerInput, "Input Layer")
            elif i == len(AllLayers) - 1:
                SNN.AddLayer(LayerOutput, "Output Layer")
            else:
                SNN.AddLayer(AllLayers[i], "Layer " + str(i))
        
        for i in range(len(AllConnections)):
            SNN.AddConnection(AllConnections[i])
            
        #SNN.Connections[1].postsynaptic_normalization(150.0)
          
        # Add the reward monitor to the nework
        #RewMonitor = RewardMonitor()
        #SNN.AddMonitor(RewMonitor, "Reward Monitor")
        return SNN
    
    def Preprocess(self, obs):
        newObs = np.asarray([
            obs[0]/4.8 if obs[0] >= 0 else 0,
            abs(obs[0])/4.8 if obs[0] < 0 else 0,
            obs[1]/4 if obs[1] >= 0 else 0,
            abs(obs[1])/4 if obs[1] < 0 else 0,
            obs[2]*2 if obs[2] >= 0 else 0,
            abs(obs[2])*2 if obs[2] < 0 else 0,
            obs[3] if obs[3] >= 0 else 0,
            abs(obs[3]) if obs[3] < 0 else 0
            #obs[4] if obs[4] >= 0 else 0,
            #abs(obs[4]) if obs[4] < 0 else 0,
            #obs[5] if obs[5] >= 0 else 0,
            #abs(obs[5]) if obs[5] < 0 else 0,
            #obs[6] if obs[6] >= 0 else 0,
            #abs(obs[6]) if obs[6] < 0 else 0,
            #obs[7] if obs[7] >= 0 else 0,
            #abs(obs[7]) if obs[7] < 0 else 0
        ], dtype=np.float32)
        return newObs
    
    def Run(self, NumEpisodesPerUpdate, NumUpdates):
        #e = 0
        #RewardTotal = 0
        AllRewards = []
        for i in range(NumUpdates):
            self.RunEpisodes(NumEpisodesPerUpdate, False)
            #RewardTotal += Reward
            #print(np.asarray(self.AllSpikesInputLayer[1][50]))
            #print(np.asarray(self.AllSpikesOutputLayer))
            
            #print(self.SpikesInputLayer)
            #print(self.SpikesOutputLayer)
            #print("Length: " + str(len(self.SpikesOutputLayer)))
            #print(self.Actions)
            #print(len(self.Rewards))
            #print(self.Steps)
            #print(sum(self.Steps))
            #print(sum(self.Steps[:-1]))
            #print(self.Network.Connections[1].w[:])
            #print(self.Network.Connections[1].w[1])
            #print(self.Network.Connections[1].w[:,0])
            #print (self.AllSpikesInputLayer[1][0].size)
            #print("Avg Fitness: " + str(np.average(self.AllTotalRewards)))
            #self.UpdateWeights()
            #self.Network.Connections[0].postsynaptic_normalization(150)
            #print(np.sum(self.Network.Connections[0].w))
            #self.Network.Connections[1].random_mutation(0.01)
            
            #e += 1
            #if (e%100 == 0):
                #print(RewardTotal / 100)
                #print (self.AllSpikesInputLayer[1][0].size)
                #print(self.Network.Connections[1].w)
                #RewardTotal = 0
            self.UpdateWeights()
            
            AllRewards.append(len(self.Rewards))
            if (i % 100 == 0) and (i != 0):
                print(sum(AllRewards[-100:])/100)
                print(self.Network.Connections[0].w)
                #print(AllRewards)
    
    def RunEpisodes(self, NumEpisodes: int, DisplayEpisode: bool):
        #self.Actions = []
        #self.AllRewards = []
        #self.Rewards = []
        self.SpikesInputLayer = []
        self.SpikesOutputLayer = []
        self.ExposuresPerStep = []
        self.Steps = []
        obs = self.Environment.reset()
        for e in range(NumEpisodes):
            obs = self.Environment.reset()
            done = False
            # Will hold all of the actions taken in the next to run episode
            self.Actions = []
            self.Rewards = []
            for _ in range(200):
                newObs = self.Preprocess(obs)
                #print(newObs)
                newObs = probabilities(newObs)
                #print(newObs)
                #print(self.Decoding)
                action, steps = self.Network.Run(newObs, self.ExposurePeriod, True, self.Decoding)
                # Add the action to the set of actions being recorded for this episode
                self.Actions.append(action)
                self.Steps.append(steps)
                #print(action)
                # Get the output spike train and make a decision based on that
                
                obs, reward, done, _ = self.Environment.step(action)
                
                self.Rewards.append(reward - 1)
                #if DisplayEpisode:
                    #self.Environment.render()
                #self.Population[AgentIndex].Monitors["Reward Monitor"].Append(reward)
                #RewardTotal += reward
                if done:
                    self.Rewards[len(self.Rewards) - 1] = -1
                    break
                
            self.SpikesInputLayer = np.asarray(self.Network.Monitors["Input Layer"].Values)
            self.SpikesOutputLayer = np.asarray(self.Network.Monitors["Output Layer"].Values)
        
            #if e == NumEpisodes - 1:
            #self.Network.Monitors["Input Layer"].Plot("Input Layer")
            self.Network.ResetStateVariables(True)
            
    def UpdateWeights(self):
        
        # Weight decay
        self.Network.Connections[0].w -= np.where(self.Network.Connections[0].w > 0, self.LearningRate/20, -1 * self.LearningRate/20)
        
        for Step in range(len(self.Rewards)):
            if (self.Rewards[Step] != 0):
                # Find the network time step that this step starts
                NetworkStepStart = sum(self.Steps[:(Step - len(self.Steps))])
                # Find the network time step that this step ends
                NetworkStepEnd = sum(self.Steps[:(Step + 1)])
                for t in range(NetworkStepEnd - NetworkStepStart):
                    for i in range(self.SpikesInputLayer[0].size):
                        if self.SpikesInputLayer[t][i] == 1:
                            Forward = 0
                            Back = 1
                            while True:
                                if (t+Forward >= NetworkStepEnd - NetworkStepStart):
                                    #print("HERE")
                                    break
                                
                                LTP = np.where(self.SpikesOutputLayer[t+Forward+NetworkStepStart] == 1, 1, 0)
                                #print(LTP)
                                if (np.sum(LTP) != 0):
                                    self.Network.Connections[0].w[:,i] += np.where(LTP == 1, self.LearningRate * self.Rewards[Step], 0)
                                else:
                                    if (t-Back < 0):
                                        break
                                    LTD = np.where(self.SpikesOutputLayer[t-Back+NetworkStepStart])
                                    if (np.sum(LTD)!=0):
                                        self.Network.Connections[0].w[:,i] -= np.where(LTD == 1, self.LearningRate * self.Rewards[Step], 0)
                                Forward += 1
                                Back += 1
            
                