# -*- coding: utf-8 -*-
import numpy as np
import sys
import gym
import math
import os

from nodes import IzhikevichNode, InputEncoder, InputNode
from connections import AbstractConnection, EncoderConnection
from network import Network
from monitors import SpikeMonitor
from encoders import probabilities, normalized_probabilities, rate_based_encoding
from decodingMethod import DecodingMethod
import matplotlib.pyplot as plt

class CartPoleRSTDP:
    
    def __init__(self, NeuronsPerLayer, ExposurePeriod: int = 50, Decoding: DecodingMethod = DecodingMethod.F2F_RESET, filename: str = "test.txt"):
        # Create the network according to the given parameters
        
        self.Environment = gym.make('CartPole-v1')
        self.Environment.reset()
        
        # Network Values
        #self.LearningRate = 1.0 # TODO: Need to recover values from hyperparameter experiment
        
        self.Decoding = Decoding
        self.ExposurePeriod = ExposurePeriod
        self.FileName = filename
        
        # Create the network
        self.Network = self.CreateNetwork(NeuronsPerLayer)
        
        self.LearningRate = 0.1
        #self.LearningRate = 0.1
        #print(self.LearningRate)
        
    def CreateNetwork(self, NeuronsPerLayer):
        
        AllLayers = []
        
        # 1: Create the layers
        for i in range(len(NeuronsPerLayer)):
            if i == 0:
                LayerInput = IzhikevichNode(NeuronsPerLayer[0])
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
                    wmin=-20.0,
                    wmax=80.0))
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
            obs[0]*0.208 if obs[0] >= 0 else 0,
            abs(obs[0])*0.208 if obs[0] < 0 else 0,
            obs[1]*0.250 if obs[1] >= 0 else 0,
            abs(obs[1])*0.250 if obs[1] < 0 else 0,
            obs[2]*2.4 if obs[2] >= 0 else 0,
            abs(obs[2])*2.4 if obs[2] < 0 else 0,
            obs[3]*0.25 if obs[3] >= 0 else 0,
            abs(obs[3])*0.25 if obs[3] < 0 else 0
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
        
    def Run(self, NumEpisodes: int = 10):
        #e = 0
        #RewardTotal = 0
        self.AllTotalRewards = []
        for i in range(NumEpisodes):
            self.AllSpikesInputLayer = []
            self.AllSpikesOutputLayer = []
            self.AllActions = []
            self.AllEpisodeRewards = []
            self.RunEpisode(False)
            #RewardTotal += Reward
            #print(np.asarray(self.AllSpikesInputLayer[1][50]))
            #print(np.asarray(self.AllSpikesOutputLayer))
            #print(self.Network.Connections[0].w)
            #print(self.Network.Connections[1].w[:])
            #print(self.Network.Connections[1].w[1])
            #print(self.Network.Connections[1].w[:,0])
            #print (self.AllSpikesInputLayer[1][0].size)
            #print("Avg Fitness: " + str(np.average(self.AllTotalRewards)))
            self.UpdateWeights()
            #self.Network.Connections[0].postsynaptic_normalization(200)
            #self.Network.Connections[0].presynaptic_normalization(100)
            self.Network.Connections[0].decay(0.005)
            #print(np.sum(self.Network.Connections[0].w))
            if (i%100 == 0 and i != 0):
                self.PrintToFile(self.FileName, self.Network.Connections[0].w)
                self.PrintToFile(self.FileName, str(sum(self.AllTotalRewards[-100:])/100))
                print(sum(self.AllTotalRewards[-100:])/100)
                print(self.Network.Connections[0].w)
            #self.Network.Connections[1].random_mutation(0.01)
            
            #e += 1
            #if (e%100 == 0):
                #print(RewardTotal / 100)
                #print (self.AllSpikesInputLayer[1][0].size)
                #print(self.Network.Connections[1].w)
                #RewardTotal = 0
        
    def RunEpisode(self, DisplayEpisode: bool):
        self.AllActions = []
        #self.AllRewards = []
        
        self.AllSpikesInputLayer = []
        self.AllSpikesOutputLayer = []
        #self.AllEpisodeRewards = []
        obs = self.Environment.reset()
        done = False
        # Will hold all of the actions taken in the next to run episode
        Actions = []
        RewardTotal = 0
        for _ in range(200):
            newObs = self.Preprocess(obs)
            #print(newObs)
            newObs = newObs*10.0
            #print(newObs)
            action, _ = self.Network.Run(newObs, self.ExposurePeriod, True, self.Decoding)
            # Add the action to the set of actions being recorded for this episode
            Actions.append(action)
            #print(action)
            # Get the output spike train and make a decision based on that
            
            obs, reward, done, _ = self.Environment.step(action)
            self.AllEpisodeRewards.append(reward)
            #if DisplayEpisode:
                #self.Environment.render()
            #self.Population[AgentIndex].Monitors["Reward Monitor"].Append(reward)
            RewardTotal += reward
            if done:
                break
            
            
        # Add all of the current actions for this episode to the AllActions list
        self.AllActions.append(Actions)
        #print(e)
        # TODO: Generalise this for multiple layers
        # Keep track of all of the spiking activity
        self.AllSpikesInputLayer.append(np.asarray(self.Network.Monitors["Input Layer"].Values))
        #print(self.Network.Monitors["Input Layer"].Values)
        self.AllSpikesOutputLayer.append(np.asarray(self.Network.Monitors["Output Layer"].Values))
        
        #if e == NumEpisodes - 1:
            #self.Network.Monitors["Input Layer"].Plot("Input Layer")
        self.Network.ResetStateVariables(True)
        
        
        self.AllTotalRewards.append(RewardTotal)
        #AvgReward = sum(self.AllTotalRewards)
        #print(AvgReward)
        
        #return (RewardTotal >= 30.0), RewardTotal

    def PrintToFile(self, TextFile, ItemToAppend):
        if os.path.exists(TextFile):
            f = open(TextFile, 'a')
            f.write(str(ItemToAppend) + "\n")
            f.close()
        else:
            f = open(TextFile, 'w')
            f.write(str(ItemToAppend) + "\n")
            f.close()
    
    def UpdateWeights(self):
        # Can retrieve the actions and the rewards from AllActions and AllTotalRewards
        
        EpisodeLength = int(len(self.AllActions[0]))
        NetworkTotalSteps = int(len(self.AllSpikesInputLayer[0]))
        #print (self.AllEpisodeRewards)
        #print (self.AllActions)
        #print(self.AllSpikesInputLayer[0])
        #self.Network.Monitors["Layer 1"]
        for t in range(len(self.AllSpikesInputLayer[0])):
            for i in range(self.AllSpikesInputLayer[0][0].size):
                
                if self.AllSpikesInputLayer[0][t][i] == 1:
                    forward = 0
                    back = 1
                    while True:
                        #print("Starting Update Weights")
                        if (t+forward >= NetworkTotalSteps or forward > 10):
                            break
                        
                        # So if the post-synaptic spike was a result of the pre-synaptic spike then set
                        # the Long Term Potentiation change to 1
                        TargetLTP = np.where(self.AllSpikesOutputLayer[0][t+forward] == 1, 1, 0)
                        #print (TargetLTP)
                        # If the TargetLTP is 1 then trigger the strengthening of the weight
                        if (np.sum(TargetLTP) != 0):
                            self.Network.Connections[0].w[:,i] += np.where(TargetLTP == 1, self.LearningRate, 0)
                            
                        else:
                            if (t-back < 0 or back > 10):
                                break
                            TargetLTD = np.where(self.AllSpikesOutputLayer[0][t-back] == 1, 1, 0)
                            if (np.sum(TargetLTD) != 0):
                                self.Network.Connections[0].w[:,i] -= np.where(TargetLTD == 1, self.LearningRate, 0)
                                
                        forward += 1
                        back += 1
                        
        
                        # If not check for LTD
                        # If LTD then trigger the weakening of the weight
                        # If neither event occured, look at one further timestep forward and back
        #self.Network.Connections[0].decay(0.05)                
'''
    def UpdateWeightsAlternate(self, Success: bool):
        NetworkTotalSteps = int(len(self.AllSpikesInputLayer[0]))
        if Success:
            for t in range(len(self.AllSpikesInputLayer[0])):
                for i in range(self.AllSpikesInputLayer[0][0].size):
                    
                    if self.AllSpikesInputLayer[0][t][i] == 1:
                        forward = 0
                        back = 1
                        while True:
                            if (t+forward >= NetworkTotalSteps or forward > 10):
                                break
                            
                            # So if the post-synaptic spike was a result of the pre-synaptic spike then set
                            # the Long Term Potentiation change to 1
                            TargetLTP = np.where(self.AllSpikesOutputLayer[0][t+forward] == 1, 1, 0)
                            
                            # If the TargetLTP is 1 then trigger the strengthening of the weight
                            if (np.sum(TargetLTP) != 0):
                                self.Network.Connections[1].w[:,i] += np.where(TargetLTP == 1, self.LearningRate * np.random.rand(), 0)
                                break
                            else:
                                if (t-back < 0 or back >= 10):
                                    break
                                TargetLTD = np.where(self.AllSpikesOutputLayer[0][t-back] == 1, 1, 0)
                                if (np.sum(TargetLTD) != 0):
                                    self.Network.Connections[1].w[:,i] -= np.where(TargetLTD == 1, self.LearningRate * np.random.rand(), 0)
                                    break
                            forward += 1
                            back += 1
        else:
            for t in range(len(self.AllSpikesInputLayer[0])):
                for i in range(self.AllSpikesInputLayer[0][0].size):
                    
                    if self.AllSpikesInputLayer[0][t][i] == 1:
                        forward = 0
                        back = 1
                        while True:
                            if (t+forward >= NetworkTotalSteps or forward > 10):
                                break
                            
                            # So if the post-synaptic spike was a result of the pre-synaptic spike then set
                            # the Long Term Potentiation change to 1
                            TargetLTP = np.where(self.AllSpikesOutputLayer[0][t+forward] == 1, 1, 0)
                            
                            # If the TargetLTP is 1 then trigger the strengthening of the weight
                            if (np.sum(TargetLTP) != 0):
                                self.Network.Connections[1].w[:,i] -= np.where(TargetLTP == 1, self.LearningRate * np.random.rand(), 0)
                                break
                            else:
                                if (t-back < 0 or back >= 10):
                                    break
                                TargetLTD = np.where(self.AllSpikesOutputLayer[0][t-back] == 1, 1, 0)
                                if (np.sum(TargetLTD) != 0):
                                    self.Network.Connections[1].w[:,i] += np.where(TargetLTD == 1, self.LearningRate * np.random.rand(), 0)
                                    break
                            forward += 1
                            back += 1'''


