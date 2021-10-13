import sys

# lunarlander_single_layer_evolution import LunarLanderSimulationEvolution
#from LunarLanderEvolution import LunarLanderEvolution

from CartPoleEvolution import CartPoleEvolution, DecodingMethod
#from CartPoleNormalizedInputEvolution import CartPoleNormalizedInputEvolution, DecodingMethod

#import decoders
#import numpy as np
#import gym

#LunarLander = LunarLanderMultilayerEvolution([14,14,32,4])

#LunarLander = LunarLanderSimulation({16,3})
#LunarLander.RunGeneration()
#LunarLander.Environment.close()



# Get the arguments form the command line
if (len(sys.argv) > 1):
    ExposurePeriod = int(sys.argv[1])
    Decoding = str(sys.argv[2])
else:
    ExposurePeriod = 50
    Decoding = "f2freset"
    
if Decoding == "f2freset":
    DecodingEnum = DecodingMethod.F2F_RESET
elif Decoding == "f2f":
    DecodingEnum = DecodingMethod.F2F
elif Decoding == "rate":
    DecodingEnum = DecodingMethod.RATE
elif Decoding == "ratereset":
    DecodingEnum = DecodingMethod.RATE_RESET
else:
    DecodingEnum = DecodingMethod.F2F_RESET
    
#print ("Trial Details: Exposure Period = " + str(ExposurePeriod) + " Decoding Method = " + str(DecodingEnum))
    
CartPole = CartPoleEvolution([8,8,2], ExposurePeriod, DecodingEnum)
#CartPole = CartPoleNormalizedInputEvolution([8,2], ExposurePeriod, DecodingEnum)

CartPole.RunGeneration()
CartPole.Environment.close()