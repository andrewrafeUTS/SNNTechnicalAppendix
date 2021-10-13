import numpy as np
import gym
import enum

from network import Network
from connections import AbstractConnection, EncoderConnection
from nodes import InputNode, IzhikevichNode, InputEncoder
from monitors import RewardMonitor
from decodingMethod import DecodingMethod
import encoders
import matplotlib.pyplot as plt

import decoders

class CartPoleEvolution():
    
    def __init__(self, NeuronsPerLayer, ExposurePeriod: int = 50, Decoding: DecodingMethod = DecodingMethod.F2F_RESET):
        # Create the network according to the number of neurons per layer
        
        self.Environment = gym.make('CartPole-v1')
        self.Environment.reset()
        
        # Population and Network Values
        self.PopulationSize = 50
        self.NumGenerations = 500
        self.ExposurePeriod = ExposurePeriod
        self.NumEpisodes = 5

        self.LearningRate = 10.0
        self.LearningRateMultiplier = 0.99
        
        # Decoding methods:
        self.Decoding = Decoding
        
        # Set up the population 
        self.Population = []
        for _ in range(self.PopulationSize):
            self.Population.append(self.CreateNetwork(NeuronsPerLayer))
            
        self.RewardTotals = []
        self.BestAgentRewardsPerGeneration = []
        
        self.TrialComplete = False
        
        print("Trial Details: Exposure Period = " + str(self.ExposurePeriod) + " Decoding Method = " + str(self.Decoding) + " Neurons Per Layer: " + str(NeuronsPerLayer))
        
    def CreateNetwork(self, NeuronsPerLayer):
        # Will create and store the network in the SNN member variable
        
        AllLayers = []
        
        # 1: Create the Layers
        for i in range(len(NeuronsPerLayer)):
            if i == 0:
                LayerInput = InputEncoder(NeuronsPerLayer[0])
                AllLayers.append(LayerInput)
            elif i == len(NeuronsPerLayer) - 1:
                LayerOutput = IzhikevichNode(NeuronsPerLayer[i])
                AllLayers.append(LayerOutput)
            else:
                AllLayers.append(IzhikevichNode(NeuronsPerLayer[i]))
        
        # 2: Add the connections between the layers
        
        AllConnections = []

        for i in range(len(AllLayers) - 1):
            if i != 0:
                AllConnections.append(AbstractConnection(
                    source=AllLayers[i],
                    target=AllLayers[i+1],
                    wmin=-20.0,
                    wmax=80.0))
            else:
                 AllConnections.append(EncoderConnection(
                    source=AllLayers[i],
                    target=AllLayers[i+1],
                    wmin=0.0,
                    wmax=150.0))
            #AllConnections[i].presynaptic_normalization(20)
            
        
        # 3: Create the network by giving the input and output layers
        SNN = Network(LayerInput, LayerOutput)
        for i in range(len(AllLayers)):
            if i == 0:
                SNN.AddLayer(LayerInput, "Input Encoder")
            elif i == len(AllLayers) - 1:
                SNN.AddLayer(LayerOutput, "Output Layer")
            else:
                SNN.AddLayer(AllLayers[i], "Layer " + str(i))
        
        for i in range(len(AllConnections)):
            SNN.AddConnection(AllConnections[i])
          
        # Add the reward monitor to the nework
        #RewMonitor = RewardMonitor()
        #SNN.AddMonitor(RewMonitor, "Reward Monitor")
        return SNN


    def Preprocess(self, obs):
        newObs = np.asarray([
            obs[0] if obs[0] >= 0 else 0,
            abs(obs[0]) if obs[0] < 0 else 0,
            obs[1] if obs[1] >= 0 else 0,
            abs(obs[1]) if obs[1] < 0 else 0,
            obs[2] if obs[2] >= 0 else 0,
            abs(obs[2]) if obs[2] < 0 else 0,
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

    '''
    def PreprocessNormalized(self, obs):
        #Preprocessing of the cart pole input layer. Uses a rescaling algorithm
        #that should be changed and tested more 
        #TODO: Try different preprocessing methods
        newObs = np.asarray([
            obs[0]/4.8 if obs[0] >= 0 else 0,
            abs(obs[0])/4.8 if obs[0] < 0 else 0,
            obs[1] if obs[1]/3 >= 0 else 0,
            abs(obs[1])/3 if obs[1] < 0 else 0,
            obs[2]/0.418 if obs[2] >= 0 else 0,
            abs(obs[2])/0.418 if obs[2] < 0 else 0,
            obs[3]/3 if obs[3] >= 0 else 0,
            abs(obs[3])/3 if obs[3] < 0 else 0
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
    '''

    def RunGeneration(self):
        for _ in range(self.NumGenerations):
            for Agent in range(self.PopulationSize):
                self.RunEpisode(Agent, self.NumEpisodes, False, False)
                #print("Completed Episode - Agent: " + str(Agent) + " - Generation: " + str(Generation))
            
            self.Evolve()
            self.LearningRate = self.LearningRate * self.LearningRateMultiplier
            #print(self.RewardTotals)
            self.RewardTotals = []
            
            if self.TrialComplete:
                break
                
    def RunEpisode(self, AgentIndex: int, NumEpisodes: int, DisplayEpisode: bool, BestAgentEpisode: bool):
        obs = self.Environment.reset()
        RewardTotal = 0
        for _ in range(NumEpisodes):
            obs = self.Environment.reset()
            done = False
            for _ in range(200):
                newObs = self.Preprocess(obs)
                #newObs = encoders.probabilities(newObs)
                action = self.Population[AgentIndex].Run(newObs, self.ExposurePeriod, True, self.Decoding)
                #print(action)
                # Get the output spike train and make a decision based on that
                
                obs, reward, done, _ = self.Environment.step(action)
                #if DisplayEpisode:
                    #self.Environment.render()
                #self.Population[AgentIndex].Monitors["Reward Monitor"].Append(reward)
                RewardTotal += reward
                if done:
                    break
                
        # Prints the average fitness over the best agent trials
        # If it is greater than the completion threshold then halt the trial
        if BestAgentEpisode:
            #print("Best Agent: Num Episodes: " + str(NumEpisodes))
            print(float(RewardTotal)/float(NumEpisodes))
            if (float(RewardTotal)/float(NumEpisodes) >= 195.0):
                self.TrialComplete = True
        
        # If it is the display episode then plot all of the monitors for inspection
        
        if DisplayEpisode:
            for MonitorName in self.Population[AgentIndex].Monitors.keys():
                self.Population[AgentIndex].Monitors[MonitorName].Plot(MonitorName)
            plt.plot(np.asarray(self.BestAgentRewardsPerGeneration))
            plt.show(block=False)
            
            '''plt.bar(x=np.arange(0, 8), height=(self.Population[AgentIndex].Connections[0].w))
            plt.title("Encoder to First")
            plt.show(block=False)
            
            plt.imshow((self.Population[AgentIndex].Connections[1].w), cmap='Greys', vmin = -20.0, vmax = 80.0)
            plt.title("First to Hidden")
            plt.show(block=False)
            
            plt.imshow((self.Population[AgentIndex].Connections[2].w), cmap='Greys', vmin = -20.0, vmax = 80.0)
            plt.title("Hidden to Output")
            plt.show(block=False)'''
            
        self.Population[AgentIndex].ResetStateVariables(True)
        
        #self.SNN.monitors["Input Spike Monitor"].plot("Input")
        #self.SNN.monitors["Output Spike Monitor"].plot("Output")
        #self.SNN.monitors["Reward Monitor"].plot("Reward")
        if not DisplayEpisode:
            self.RewardTotals.append(RewardTotal)

        # Reset all the node variables

            
    def Evolve(self):
        Percentage = int(len(self.Population)*0.2)
        AgentFitness = np.asarray(self.RewardTotals)
        
        BestAgent = np.argpartition(AgentFitness, -1)[-1:]
        BestAgentReward = max(self.RewardTotals)
        # Only rerun the 100 trials if the average reward for the three trials
        # was greater than the 10
        if (float(BestAgentReward)/float(self.NumEpisodes) >= 195.0):
            self.RunEpisode(BestAgent[0], 100, False, True)
            # Don't go through the final evolution if the trial is already complete
            if (self.TrialComplete):
                print(self.Population[BestAgent[0]].Connections[0].w)
                print(self.Population[BestAgent[0]].Connections[1].w)
                return
        else:
            print(float(BestAgentReward)/float(self.NumEpisodes))
            
        self.BestAgentRewardsPerGeneration.append(self.RewardTotals[BestAgent[0]])
        #print(BestAgent)
        #print(self.Population[BestAgent[0]].Connections[0].w)
        #print(self.Population[BestAgent[0]].Connections[1].w)
        
        
        BestAgentFitness = np.argpartition(AgentFitness, -Percentage)[-Percentage:]
        
        for i in range(len(self.Population)):
            if i not in BestAgentFitness:
                # Find two of the best agents and use them for crossover
                Agent1 = BestAgentFitness[
                    np.random.randint(low=0, high=len(BestAgentFitness))]
                Agent2 = BestAgentFitness[
                    np.random.randint(low=0, high=len(BestAgentFitness))]
                
                # Create a set of true false random choices to choose which
                # agent should be used for the crossover of each weight
                
                # Loop through each of the connections and crossover or merge the weights
                for j in range(len(self.Population[i].Connections)):
                    
                    AgentChoices = np.random.choice(
                        [True, False],
                        self.Population[i].Connections[j].w.shape)
                
                    self.Population[i].Connections[j].w = np.where(
                            AgentChoices,
                            self.Population[Agent1].Connections[j].w,
                            self.Population[Agent2].Connections[j].w)
                            
                    # Will change the weight based on some random mutation chance.
                    
                    MutationChance = np.random.random(self.Population[i].Connections[j].w.shape)
                    
                    self.Population[i].Connections[j].w = np.where(np.logical_and(MutationChance >= 0.0, MutationChance < 0.01), self.Population[i].Connections[j].w + self.LearningRate * 10, self.Population[i].Connections[j].w)
                    self.Population[i].Connections[j].w = np.where(np.logical_and(MutationChance >= 0.01, MutationChance < 0.02), self.Population[i].Connections[j].w - self.LearningRate * 10, self.Population[i].Connections[j].w)
                    self.Population[i].Connections[j].w = np.where(np.logical_and(MutationChance >= 0.02, MutationChance < 0.12) , self.Population[i].Connections[j].w + self.LearningRate, self.Population[i].Connections[j].w)
                    self.Population[i].Connections[j].w = np.where(np.logical_and(MutationChance >= 0.12, MutationChance < 0.22), self.Population[i].Connections[j].w - self.LearningRate, self.Population[i].Connections[j].w)
                    
                    #self.Population[i].Connections[j].presynaptic_normalization(10.0)
                    
        
        
        
        
        
        
        
        
        