from nodes import AbstractNode
from connections import AbstractConnection
from monitors import AbstractMonitor, SpikeMonitor
from decodingMethod import DecodingMethod
from encoders import probabilities
import numpy as np
import decoders

class Network():

    ''' Constructor for the Network class.
        @param InputLayer - The input layer of the network
        @param OutputLayer - The output layer of the network
    '''
    def __init__(self, InputLayer, OutputLayer):
        
        # Set up the network with empty dictionaries
        self.Layers = {}
        self.LTP = []
        self.Connections = []
        self.Monitors = {}
        # Assigne the input and output layers
        self.InputLayer = InputLayer
        self.OutputLayer = OutputLayer
        
        self.AllVUpdate = []
        
        # Create the spike monitors for the input and output layer and add them
        # to the network.
        #OutputSpikeMonitor = SpikeMonitor(self.OutputLayer)
        #InputSpikeMonitor = SpikeMonitor(self.InputLayer)
        #self.AddMonitor(OutputSpikeMonitor, "Output Spike Monitor")
        #self.AddMonitor(InputSpikeMonitor, "Input Spike Monitor")
    
    ''' Will add a layer of nodes to the network.
        @param Layer - An AbstractNode layer to be added to the network
        @param Name: The name of the layer in the network
    '''
    def AddLayer(self, Layer: AbstractNode, Name: str) -> None:
        # Check that the layer exists and that the name is not empty
        assert(
            Layer is not None and Name != ""
        ), 'The layer must exist and the layer name must not be an empty string'

        self.Layers[Name] = Layer
        if Name != "Input Encoder":
            LayerSpikeMonitor = SpikeMonitor(Layer)
            self.AddMonitor(LayerSpikeMonitor, Name)
    
    ''' Will add a connection between two layers to the network.
        @param Connection - An abstract connection between two layers 
    '''
    def AddConnection(self, Connection: AbstractConnection) -> None:
        # Check that the connection exists
        assert(
            Connection is not None
        ), 'The connection must not be None'
        # Adds the connection to the connections source layer, i.e.
        # source layer connects to target layer
        #self.Connections[Connection.source] = Connection
        self.Connections.append(Connection)
        self.LTP.append(np.zeros(Connection.w.shape))
        #print(self.LTP)
    
    ''' Will add a monitor to the network.
        @param Monitor - An AbstractMonitor that monitors a layer in the network
    '''
    def AddMonitor(self, Monitor: AbstractMonitor, Name: str) -> None:
        # Make sure that the monitor is not none
        assert(
            Monitor is not None
        ), 'The monitor must not be none'
        
        # Add the monitor to the dictionary of monitors.
        self.Monitors[Name] = Monitor
    
    ''' Runs a single iteration of the network.
        @param Inputs - The preprocessed set of Inputs into the network. 
                        Must be the same length as the input layer.
    '''
    def RunSingle(self, Inputs):
        CurrentLayer = self.InputLayer
        #x = probabilities(Inputs)
        ############################# CHANGED FOR THE RSTDP METHOD ############################
        x = Inputs # THE METHOD USED WITH INPUT ENCODER
        # Pass the inputs as the injection into the first connection
        j = 0
        
        #AllV = []
        
        while (CurrentLayer != self.OutputLayer):
            # Process layer by layer, injecting the spike train produced
            # into the next layer.
            if j > 0:
                self.LTP[j] = np.multiply(self.Connections[j-1].plasticity.transpose(), self.Connections[j-1].target.Spikes).transpose()
                
            self.Connections[j].process(x)
            x = self.Connections[j].x
            CurrentLayer = self.Connections[j].target
            j += 1
         
        self.LTP.append(np.multiply(self.Connections[j-1].plasticity.transpose(), self.Connections[j-1].target.Spikes).transpose())
           
            
        #print(self.LTP)
        # At this point we have an x which is to be injected into the 
        # output layer then the output layer contains a spike vector of 
        # the current spiking activity and can be used for action selection
        self.OutputLayer.Process(x)
        self.Spikes = self.OutputLayer.Spikes
        # Then update all the monitors for the current network step
        self.UpdateMonitors()
        
        
        #return AllV
        
        
    ''' Runs one timestep of the network (the number of single iterations in
        the exposure period)
        @param Inputs - The inputs to the network. Must be the same length as
                        the input layer.
        @param ExposurePeriod - The number of individual cycles in one timestep
        @param IsCartpole - A boolean representing if it is cartpole or 
                            lunar lander
        @return SpikeTrain - The spike train of the output layer over the
                             exposure period.
    '''
    def Run(self, Inputs, ExposurePeriod, IsCartpole, Decoding):
        # Run the single process of the entire network for each step in the
        # exposure period of the stimulus.
        #print(Decoding == DecodingMethod.F2F or DecodingMethod.F2F_RESET)
        if IsCartpole:
            if Decoding == DecodingMethod.F2F or Decoding == DecodingMethod.F2F_RESET:
                for i in range(ExposurePeriod):
                    self.RunSingle(Inputs)
                    action = decoders.first_to_spike(self.GetOutputSpikeTrain(1), IsCartpole)
        
                    if action != -1:
                        if Decoding == DecodingMethod.F2F_RESET:
                            self.ResetStateVariables(False)
                        return action, i + 1
                    
            elif Decoding == DecodingMethod.RATE or Decoding == DecodingMethod.RATE_RESET:
                for i in range(ExposurePeriod):
                    self.RunSingle(Inputs)
                    
                action = decoders.select_max_spiking_cartpole(self.GetOutputSpikeTrain(ExposurePeriod))
                if action != -1:
                    if Decoding == DecodingMethod.RATE_RESET:
                        self.ResetStateVariables(False)
                    #print("Selected Action: " + str(action))
                    return action, ExposurePeriod
            
            # Otherwise return a random move
            #print("RANDOM MOVE")
            return np.random.randint(low=0, high=2), ExposurePeriod
        
        else: # Is Lunar Lander
            if Decoding == DecodingMethod.F2F or Decoding == DecodingMethod.F2F_RESET:
                for i in range(ExposurePeriod):
                    self.RunSingle(Inputs)
                    action = decoders.first_to_spike(self.GetOutputSpikeTrain(1), IsCartpole)
        
                    if action != -1:
                        if Decoding == DecodingMethod.F2F_RESET:
                            self.ResetStateVariables(False)
                        return action, i + 1
                    
            elif Decoding == DecodingMethod.RATE or Decoding == DecodingMethod.RATE_RESET:
                for i in range(ExposurePeriod):
                    self.RunSingle(Inputs)
                    
                action = decoders.select_max_spiking_lunar_lander(self.GetOutputSpikeTrain(ExposurePeriod))
                if action != -1:
                    if Decoding == DecodingMethod.RATE_RESET:
                        self.ResetStateVariables(False)
                    return action, ExposurePeriod
                
            # Otherwise return a random move
            return np.random.randint(low=0, high=4), ExposurePeriod
                
        
        ''' Old code
        for i in range(ExposurePeriod):
            self.RunSingle(Inputs)
            action = decoders.first_to_spike(self.GetOutputSpikeTrain(1), IsCartpole)

            if action != -1:
                self.ResetStateVariables(False)
                return action
        # Find the highest spiking output neuron and return the action back
        #print(self.get_output_spike_train(exposure_period))
        
        # IMPORTANT - THIS IS WHERE THE DECODING METHOD IS SELECTED!!!
        
        if IsCartpole:
            return np.random.randint(low=0, high=2)
            #return decoders.select_max_spiking_cartpole(
            #    self.GetOutputSpikeTrain(ExposurePeriod))
        else:
            return decoders.select_max_spiking_lunar_lander(
                self.GetOutputSpikeTrain(ExposurePeriod))
        '''
        
    ''' Retrieves the spike train of the output layer over the exposure period.
        @param ExposurePeriod - The number of individual cycles in one timestep
        @return SpikeTrain - The spike train of the output layer.
    '''
    def GetOutputSpikeTrain(self, ExposurePeriod):
        return self.Monitors["Output Layer"].GetSpikeTrain(
            ExposurePeriod)

    ''' Adds the spike trains of the single iteration to the monitors.
    '''
    def UpdateMonitors(self) -> None:
        # Loop through each of the monitors and update the spike trains for the
        # current single time step.
        for name in self.Monitors.keys():
            if isinstance(self.Monitors[name], SpikeMonitor):
                self.Monitors[name].Append(self.Monitors[name].Source.Spikes)
    
    
    ''' Resets all of the layers and monitors back to their defaults.
    '''
    def ResetStateVariables(self, ShouldResetMonitors) -> None:
        # Loop through all the layers and reset neurons to default values
        for LayerName in self.Layers:
            self.Layers[LayerName].Reset()
        
        for Connection in self.Connections:
            Connection.reset_state_variables()
            
        self.LTP = []
        #print(LTP)
            
        if ShouldResetMonitors:
            # Loop through all monitors and reset them to default values
            for MonitorName in self.Monitors.keys():
                self.Monitors[MonitorName].ResetStateVariables()
                
            

        
    
