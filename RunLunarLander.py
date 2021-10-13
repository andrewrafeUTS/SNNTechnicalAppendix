import sys

from LunarLanderEvolution import LunarLanderEvolution
from decodingMethod import DecodingMethod

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
    
LunarLander = LunarLanderEvolution([14,14,4], ExposurePeriod, DecodingEnum)
LunarLander.RunGeneration()
LunarLander.Environment.close()
