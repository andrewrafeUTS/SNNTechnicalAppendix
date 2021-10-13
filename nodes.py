from abc import ABC, abstractmethod
import numpy as np

''' The abstract node class which holds an entire layer of nodes.
'''
class AbstractNode():

    ''' Constructor for the abstract node class
        @param NumNeuronsInLayer - The number of neurons in the layer
    '''
    def __init__(self, NumNeuronsInLayer: int) -> None:
        self.NumNeuronsInLayer = NumNeuronsInLayer
        self.Spikes = np.zeros(NumNeuronsInLayer)
        self.OverflowCount = 0

    ''' Will process the layer of neurons with given injections for a single 
        step.
        @param Injection - The injections into the layer
    '''
    @abstractmethod
    def Process(self, Injection):
        # Check that the size of x is equal to the size of the layer
        #assert(Injection.size == self.NumNeuronsInLayer)
        pass
    
    ''' Will reset the spikes of this layer of nodes.
    '''
    @abstractmethod
    def Reset(self):
        self.Spikes.fill(0)
    
class InputEncoder(AbstractNode):

    def __init__(self, NumNeuronsInLayer: int) -> None:
        super().__init__(NumNeuronsInLayer)
        
    def Process(self, Injection) -> None:
        super().Process(Injection)
        self.Spikes = Injection
        
    def Reset(self):
        super().Reset()

''' The input node class which holds an entire layer of input neurons
'''
class InputNode(AbstractNode):

    def __init__(self, NumNeuronsInLayer: int) -> None:
        super().__init__(NumNeuronsInLayer)

    def Process(self, Injection) -> None:
        super().Process(Injection)
        # Simply apply the given spikes to this nodes spike train
        self.Spikes = Injection
    
    def Reset(self):
        super().Reset()

''' The Izhikevich node class which holds an entire layer of regular spiking
    izhikevich neurons.
'''
class IzhikevichNode(AbstractNode):
    
    def __init__(self, NumNeuronsInLayer: int) -> None:
        super().__init__(NumNeuronsInLayer)
        # Apply all of the izhikevich constants
        self.A = 0.02
        self.B = 0.2
        self.C = -65.0
        self.D = 8
        self.Peak = 30.0
        self.Rest = -60
        self.Thresh = -40.0

        self.V = np.zeros(self.NumNeuronsInLayer)
        self.V.fill(self.Rest)
        self.U = np.zeros(self.NumNeuronsInLayer)
        self.U.fill(self.B*self.Rest)
    
    def Process(self, Injection) -> None:
        super().Process(Injection)
        
        # IF IT IS A NEGATIVE INJECTION THEN ASSUME THAT IT WAS UNABLE TO 
        # TRANSMIT TO NEXT NEURON i.e. INJECTION OF ZERO
        Injection = np.where(Injection < 0, 0, Injection)
        # Apply the neuron updates with the x being the injections
        # Split the updates to half steps to prevent explosion
        # Ensure that x is not too big
        # x = np.where(x > 100, 100, x)
        # x = np.where(x < -70, -70, x)
        try:
            OldV = self.V
            OldU = self.U
            self.V += 0.5 * (0.04 * self.V**2 + 5*self.V + 140 - self.U + Injection)
            self.V += 0.5 * (0.04 * self.V**2 + 5*self.V + 140 - self.U + Injection)
            self.U += self.A*(self.B*self.V - self.U)
        except FloatingPointError as _:
            #print('Overflow with old v: ', old_v, err)
            self.OverflowCount += 1
            self.V = np.where(OldV >= 100, 100, OldV)
            self.V = np.where(OldV <= -100, self.C, OldV)
            self.U = OldU
            if self.OverflowCount == 1:
                print('Weights have become too big')
                #assert(False)
            #print('Setting v to ', self.v)

        # Check for any neuron spikes and add them to the spike list
        self.Spikes = np.where(self.V >= self.Peak, 1, 0)

        # Reset those neurons that spiked
        self.V = np.where(self.Spikes >= 1, self.C, self.V)
        self.U = np.where(self.Spikes >= 1, self.U + self.D, self.U)

        # At the conclusion of this method, the self.s spikes will be populated with the
        # appropriate spiking neurons and those neurons reset according to the above rules
        return self.V
    
    def Reset(self):
        super().Reset()
        self.V.fill(self.Rest)
        self.U.fill(0)
    


    
