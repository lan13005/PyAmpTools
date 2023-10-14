import ROOT
from ROOT import pythonization
import os

USE_MPI = os.environ['ATI_USE_MPI'] == "1" if 'ATI_USE_MPI' in os.environ else False
USE_GPU = os.environ['ATI_USE_GPU'] == "1" if 'ATI_USE_GPU' in os.environ else False
USE_GPU = False
RANK_MPI = int(os.environ['ATI_RANK']) if 'ATI_RANK' in os.environ else 0
SUFFIX  = "_GPU" if USE_GPU else ""
SUFFIX += "_MPI" if USE_MPI else ""
print("\n------------------------------------------------")
print(f'MPI is {"enabled" if USE_MPI else "disabled"}')
print(f'GPU is {"enabled" if USE_GPU else "disabled"}')


#################### LOAD LIBRARIES (ORDER MATTERS!) ###################
if RANK_MPI == 0: print(f'Loading library libAmpTools{SUFFIX}.so.....', end='')
ROOT.gSystem.Load(f'libAmpTools{SUFFIX}.so')
if RANK_MPI == 0: print(f'  *** Loaded\nLoading library libAmpPlotter.so...', end='')
ROOT.gSystem.Load('libAmpPlotter.so')
if RANK_MPI == 0: print(f'  *** Loaded\nLoading library libAmpsDataIO{SUFFIX}.so...', end='')
ROOT.gSystem.Load(f'libAmpsDataIO{SUFFIX}.so')
if RANK_MPI == 0: print(f'  *** Loaded')
print("------------------------------------------------\n")

# Dummy functions that just prints "initialization"
#  This is to make sure the libraries are loaded
#  as python is interpreted.
ROOT.initialize( RANK_MPI == 0 )

##################### SET ALIAS ########################
gInterpreter                = ROOT.gInterpreter
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
AmplitudeInfo               = ROOT.AmplitudeInfo

############## UTILITY ##############
def raiseError(errorType, msg):
    raise errorType(msg)

############## PYTHONIZATION ##############
def _parMgr_returnPar_if_keyExists(self,key):
    if not self.findParameter(key):
        raiseError(KeyError, f'Parameter {key} not found by ParameterManager')
    return self.findParameter(key)

# Pythonize requires v6.26 or later
@pythonization("ParameterManager")
def pythonize_parMgr(klass):
    klass.__repr__ = lambda self: '\n'.join([f'{k}: {self.findParameter(k).value()}' for k in self.getProdParList()])
    klass.__len__  = lambda self: self.getProdParList().size()
    klass.__getitem__ = lambda self, key: _parMgr_returnPar_if_keyExists(self,key).value()
    klass.__setitem__ = lambda self, key, value: _parMgr_returnPar_if_keyExists(self,key).setValue(value)
    klass.__contains__ = lambda self, key: key in self.getProdParList()
