from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from nodes import AbstractNode

''' The abstract class for monitors
'''
class AbstractMonitor():

    ''' Default construct for abstract monitor class.
        @param Source - The AbstractNode layer that the monitor is attached to
    '''
    def __init__(self, Source: AbstractNode) -> None:
        self.Source = Source
        self.Values = []

    ''' Will plot the function using matplotlib pyplot
    '''
    @abstractmethod
    def Plot(self) -> None:
        pass

    ''' Will append a value to the list of values on this monitor
        @param Value - The value to be added to the list of values on the
                       monitor.
    '''
    @abstractmethod
    def Append(self, value) -> None:
        pass

''' The monitor class to store spike trains of a single layer
'''
class SpikeMonitor(AbstractMonitor):

    def __init__(self, Source: AbstractNode) -> None:
        super().__init__(Source)
    
    def Plot(self, Title: str) -> None:
        x = []
        y = []
        # Convert the values into a numpy array and plot them as a scatter
        for i in range(len(self.Values)):
            for j in range(len(self.Values[i])):
                # If it is a 1 then add the i and j value to the x and y lists
                if self.Values[i][j] == 1:
                    x.append(i)
                    y.append(j)
        plt.scatter(x,y,s=1)
        plt.title(label=Title)
        print("Showing Plot")
        plt.show()
    
    def Append(self, Value) -> None:
        self.Values.append(np.copy(Value))
    
    def ResetStateVariables(self) -> None:
        self.Values = []
        
    def GetSpikeTrain(self, ExposurePeriod):
        # Returns the spike train of the most recent additions
        # to the monitor with the length of the exposure period.
        return self.Values[-ExposurePeriod:]
    
''' The monitor class to store the rewards achieved at each time step
'''
class RewardMonitor(AbstractMonitor):
    
    def __init__(self) -> None:
        super().__init__(None)
        
    def Append(self, Value) -> None:
        self.Values.append(np.copy(Value))
        
    def ResetStateVariables(self) -> None:
        self.Values = []
        
    def Plot(self, Title: str) -> None:
        plt.plot(self.Values)
        plt.title(label=Title)
        plt.show()
        
class NeuronPotentialMonitor(AbstractMonitor):
    
    def __init__(self) -> None:
        super().__init__(None)
        
    def Append(self, Value) -> None:
        self.Values.append(np.copy(Value))
        
    def ResetStateVariables(self) -> None:
        self.Values = []
        
    def Plot(self, Title: str) -> None:
        plt.plot(self.Values)
        plt.title(label=Title)
        plt.show()
        
class InputEncoderMonitor(AbstractMonitor):
    
    def __init__(self) -> None:
        super().__init__(None)
        
    def Append(self, Value) -> None:
        self.Values.append(np.copy(Value))
        
    def ResetStateVariables(self) -> None:
        self.Values = []
        
    def Plot (self, Title: str) -> None:
        plt.plot(self.Values)
        plt.title(label=Title)
        plt.show()

