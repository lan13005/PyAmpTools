import ROOT
import os
from typing import List

############## SET ENVIRONMENT VARIABLES ##############
REPO_HOME     = os.environ['REPO_HOME']

############## LOAD LIBRARIES ##############
ROOT.gSystem.Load('libAmps.so')
ROOT.gSystem.Load('libDataIO.so')
ROOT.gSystem.Load('libAmpTools.so')

# Dummy functions that just prints initialization
#  This is to make sure the libraries are loaded
#  as python is interpreted
ROOT.initializeAmps()
ROOT.initializeDataIO()

################ SET ALIAS ###################
ConfigFileParser  = ROOT.ConfigFileParser
ConfigurationInfo = ROOT.ConfigurationInfo
AmpToolsInterface = ROOT.AmpToolsInterface
Zlm               = ROOT.Zlm
ROOTDataReader    = ROOT.ROOTDataReader
ParameterManager  = ROOT.ParameterManager


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


############## Utility Functions ##############
class Params:
    ''' Struct to hold parameters '''
    def __init__(self, name, value):
        self.name : str     = name
        self.value: complex = value

    def __repr__(self):
        return f'{self.name} = {self.value}'

def getNLL(
    params_update: List[Params] = []
    ):
    nll = 1e7
    for params in params_update:
        print(f'Setting {params.name} to {params.value}')
        parMgr.setProductionParameter(params.name, params.value)
    nll = ati.likelihood()
    print(f'Negative LogLikelihood: {nll}')
    return nll

############## EXTRACT LIKELIHOODS #############
getNLL()
params_update = [
     Params("etapi::reZ::resAmp1", 5+4j),
]
getNLL(params_update)
