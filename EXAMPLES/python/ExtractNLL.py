import ROOT
import os
from typing import List

############## SET ENVIRONMENT VARIABLES ##############
REPO_HOME     = os.environ['REPO_HOME']
os.environ['ATI_USE_MPI'] = "1" # set to 1 to use MPI libraries
from atiSetup import *

############## LOAD CONFIGURATION FILE ##############
cfgfile = f'{REPO_HOME}/gen_amp/fit_res.cfg'
parser = ConfigFileParser(cfgfile)
cfgInfo: ConfigurationInfo = parser.getConfigurationInfo()
cfgInfo.display()

############## REGISTER OBJECTS FOR AMPTOOLS ##############
AmpToolsInterface.registerAmplitude( Zlm() )
AmpToolsInterface.registerDataReader( ROOTDataReader() )

ati = AmpToolsInterface( cfgInfo )
parMgr: ParameterManager = ati.parameterManager()

print("Parameters:\n-----------------")
print(f'Number of params: {len(parMgr)}')
print(parMgr)
print(f'Initial Likelihood {ati.likelihood()}')
key = 'etapi::imZ::resAmp1'
print(f'Is {key} in parMgr? {key in parMgr}')
parMgr[key] = complex(15,10)
print(parMgr)
print(f'Final Likelihood {ati.likelihood()}')
