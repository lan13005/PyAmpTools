import ROOT
from ROOT import pythonization
import os

def checkEnvironment(variable):
    ''' Check if environment variable is set to 1 '''
    return os.environ[variable] == "1" if variable in os.environ else False

def loadLibrary(libName, RANK_MPI, availability=True):
    ''' Load shared library and print availability '''
    statement = f'Loading library {libName} '
    if RANK_MPI == 0: print(f'{statement:.<35}', end='')
    if availability: ROOT.gSystem.Load(libName)
    status = "ON" if availability else "OFF"
    if RANK_MPI == 0: print(f' {status}')

USE_MPI = checkEnvironment('ATI_USE_MPI')
USE_GPU = checkEnvironment('ATI_USE_GPU')
RANK_MPI = int(os.environ['ATI_RANK']) if 'ATI_RANK' in os.environ else 0
SUFFIX  = "_GPU" if USE_GPU else ""
SUFFIX += "_MPI" if USE_MPI else ""

print("\n------------------------------------------------")
print(f'MPI is {"enabled" if USE_MPI else "disabled"}')
print(f'GPU is {"enabled" if USE_GPU else "disabled"}')
#################### LOAD LIBRARIES (ORDER MATTERS!) ###################
USE_FSROOT = checkEnvironment('ATI_USE_FSROOT')
loadLibrary(f'libAmpTools{SUFFIX}.so', RANK_MPI)
loadLibrary(f'libAmpPlotter.so', RANK_MPI)
loadLibrary(f'libAmpsDataIO{SUFFIX}.so', RANK_MPI) # Depends on AmpPlotter!
loadLibrary(f'libFSRoot.so', RANK_MPI, USE_FSROOT)
print("------------------------------------------------\n")

# Dummy functions that just prints "initialization"
#  This is to make sure the libraries are loaded
#  as python is interpreted.
ROOT.initialize( RANK_MPI == 0 )
if USE_FSROOT: ROOT.initialize_fsroot( RANK_MPI == 0 )

##################### SET ALIAS ########################
gInterpreter                = ROOT.gInterpreter
ConfigFileParser            = ROOT.ConfigFileParser
ConfigurationInfo           = ROOT.ConfigurationInfo
if USE_MPI:
    DataReader              = ROOT.DataReaderMPI['ROOTDataReader'] # DataReaderMPI is a template; use [] to specify the type
    DataReaderFilter        = ROOT.DataReaderMPI['ROOTDataReaderFilter']
    AmpToolsInterface       = ROOT.AmpToolsInterfaceMPI
else:
    DataReader              = ROOT.ROOTDataReader
    DataReaderFilter        = ROOT.ROOTDataReaderFilter
    AmpToolsInterface       = ROOT.AmpToolsInterface
Zlm                         = ROOT.Zlm
BreitWigner                 = ROOT.BreitWigner
Piecewise                   = ROOT.Piecewise
PhaseOffset                 = ROOT.PhaseOffset
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
