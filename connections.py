from abc import ABC, abstractmethod
import numpy as np

from nodes import AbstractNode

class AbstractConnection():

    def __init__(self, source: AbstractNode, target: AbstractNode, wmin: float = 0, wmax: float = 1) -> None:
        # Check to make sure that both a source and a target exist
        assert(
            source is not None and target is not None
        ), 'Both the source and the target must not be None'
        # Set the source and the target member variables
        self.source = source
        self.target = target
        self.wmin = wmin
        self.wmax = wmax
        self.w = self.wmin + (np.random.random_sample(size=(target.NumNeuronsInLayer, source.NumNeuronsInLayer)) * self.wmax)
        #self.presynaptic_normalization(100.0)
    
    @abstractmethod
    def process(self, x) -> None:
        # Pass the x to the source layer to produce the spikes
        self.source.Process(x)
        #s = np.transpose(self.source.s)
        s = self.source.Spikes
        # Create the injection to the target neurons through multiply weight matrix
        # by the spike vector produced by the source layer
        self.x = np.dot(self.w, s)
        # This self.x is ready to be passed to the next connection in the network
        # If this is the final connection then need to feed this into the final layer
        # to produce the output spikes

    @abstractmethod
    def reset_state_variables(self) -> None:
        pass

    def presynaptic_normalization(self, norm: float):
        """ Will make sure that the outgoing weights from each source neuron when summed equal the norm amount.
        """
        # Will be essentially the same as postsynaptic neurons except will operate on different axis
        #print(self.w)
        # Clamp to the wmin and wmax
        self.w = np.clip(self.w, a_min = self.wmin, a_max = self.wmax)
        sum_matrix = np.tile(self.w.sum(axis=0),(self.target.NumNeuronsInLayer,1))
        #print(sum_matrix)
        self.w = self.w/sum_matrix * norm
        #self.w = np.divide(self.w.T, self.w.sum(axis=0)[:,np.newaxis]).T * norm
        #print(f"After {self.w}")
        #print(self.w.sum(axis=0))
        #print(self.w)

    def postsynaptic_normalization(self, norm: float = 1):
        """ Will make sure that the incoming weights to the target neuron when summed equal the norm amount.
        """
        #print("NOT PROPERLY IMPLEMENTED POSTSYNAPTIC NORMALIZATION")
        self.w = np.transpose((np.transpose(self.w) / self.w.sum(axis=1)) * norm)
        #print(self.w)
        self.w = np.clip(self.w, a_min = self.wmin, a_max = self.wmax)
        
class EncoderConnection(AbstractConnection):
    
    def __init__(self, source: AbstractNode, target: AbstractNode, wmin: float = 0, wmax: float = 1) -> None:
        assert(source is not None and target is not None), 'Both the source and the target must not be None'
        
        self.source = source
        self.target = target
        self.wmin = wmin
        self.wmax = wmax
        self.w = self.wmin + (np.random.random(size=(self.target.NumNeuronsInLayer)) * self.wmax)
        #print(self.w)
        
    def process(self, x) -> None:
        #super().process(x)
        self.source.Process(x)
        s=self.source.Spikes
        self.x = self.w * s
        
    def reset_state_variables(self) -> None:
        pass