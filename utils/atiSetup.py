import ROOT
from ROOT import pythonization
import os

USE_MPI = os.environ['ATI_USE_MPI'] == "1" if 'ATI_USE_MPI' in os.environ else False
RANK_MPI = int(os.environ['ATI_RANK']) if 'ATI_RANK' in os.environ else None
SUFFIX = "_MPI" if USE_MPI else ""


#################### LOAD LIBRARIES ###################
ROOT.gSystem.Load(f'libAmpTools{SUFFIX}.so')
ROOT.gSystem.Load(f'libDataIO.so')
ROOT.gSystem.Load(f'libAmps.so')
ROOT.gSystem.Load('libAmpPlotter.so')

if RANK_MPI == 0:
    print(f'Loaded libraries: libAmpTools{SUFFIX}.so, libDataIO.so, libAmps.so')

# Dummy functions that just prints "initialization"
#  This is to make sure the libraries are loaded
#  as python is interpreted.
ROOT.initializeAmps(   RANK_MPI == 0 )
ROOT.initializeDataIO( RANK_MPI == 0 )

##################### SET ALIAS ########################
ConfigFileParser            = ROOT.ConfigFileParser
ConfigurationInfo           = ROOT.ConfigurationInfo
if USE_MPI:
    DataReader              = ROOT.DataReaderMPI['ROOTDataReader'] # DataReaderMPI is a template; use [] to specify the type
    AmpToolsInterface       = ROOT.AmpToolsInterfaceMPI
else:
    DataReader              = ROOT.ROOTDataReader
    AmpToolsInterface       = ROOT.AmpToolsInterface
Zlm                         = ROOT.Zlm
ParameterManager            = ROOT.ParameterManager
MinuitMinimizationManager   = ROOT.MinuitMinimizationManager
########### PLOTTER / RESULTS RELATED ###########
FitResults                  = ROOT.FitResults
EtaPiPlotGenerator          = ROOT.EtaPiPlotGenerator
PlotGenerator               = ROOT.PlotGenerator
TH1                         = ROOT.TH1
TFile                       = ROOT.TFile

############## LOAD LIBRARIES ##############
ROOT.gSystem.Load('libAmps.so')
ROOT.gSystem.Load('libDataIO.so')
ROOT.gSystem.Load('libAmpTools.so')

# Dummy functions that just prints initialization
#  This is to make sure the libraries are loaded
#  as python is interpreted
ROOT.initializeAmps(True)
ROOT.initializeDataIO(True)

################ SET ALIAS ###################
ConfigFileParser            = ROOT.ConfigFileParser
ConfigurationInfo           = ROOT.ConfigurationInfo
AmpToolsInterface           = ROOT.AmpToolsInterface
Zlm                         = ROOT.Zlm
ROOTDataReader              = ROOT.ROOTDataReader
ParameterManager            = ROOT.ParameterManager
AmplitudeInfo               = ROOT.AmplitudeInfo
MinuitMinimizationManager   = ROOT.MinuitMinimizationManager

############## UTILITY ##############
def raiseError(errorType, msg):
    raise errorType(msg)

############## PYTHONIZATION ##############
def _parMgr_returnPar_if_keyExists(self,key):
    if not self.findParameter(key):
        raiseError(KeyError, f'Parameter {key} not found by ParameterManager')
    return self.findParameter(key)

@pythonization("ParameterManager")
def pythonize_parMgr(klass):
    klass.__repr__ = lambda self: '\n'.join([f'{k}: {self.findParameter(k).value()}' for k in self.getProdParList()])
    klass.__len__  = lambda self: self.getProdParList().size()
    klass.__getitem__ = lambda self, key: _parMgr_returnPar_if_keyExists(self,key).value()
    klass.__setitem__ = lambda self, key, value: _parMgr_returnPar_if_keyExists(self,key).setValue(value)
    klass.__contains__ = lambda self, key: key in self.getProdParList()
