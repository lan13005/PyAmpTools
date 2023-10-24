import ROOT
from ROOT import pythonization
import os
import cppyy

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
loadLibrary(f'libAmpsGen.so', RANK_MPI)
print("------------------------------------------------\n")

# Dummy functions that just prints "initialization"
#  This is to make sure the libraries are loaded
#  as python is interpreted.
ROOT.initialize( RANK_MPI == 0 )
ROOT.initialize_gen( RANK_MPI == 0 )

gInterpreter                = ROOT.gInterpreter
ConfigFileParser            = ROOT.ConfigFileParser
ConfigurationInfo           = ROOT.ConfigurationInfo
if USE_MPI:
    DataReader              = ROOT.DataReaderMPI['ROOTDataReader'] # DataReaderMPI is a template; use [] to specify the type
    DataReaderFilter        = ROOT.DataReaderMPI['ROOTDataReaderFilter']
    DataReaderBootstrap     = ROOT.DataReaderMPI['ROOTDataReaderBootstrap']
    AmpToolsInterface       = ROOT.AmpToolsInterfaceMPI
else:
    DataReader              = ROOT.ROOTDataReader
    DataReaderFilter        = ROOT.ROOTDataReaderFilter
    DataReaderBootstrap     = ROOT.ROOTDataReaderBootstrap
    AmpToolsInterface       = ROOT.AmpToolsInterface
Zlm                         = ROOT.Zlm
BreitWigner                 = ROOT.BreitWigner
Piecewise                   = ROOT.Piecewise
PhaseOffset                 = ROOT.PhaseOffset
TwoPiAngles                 = ROOT.TwoPiAngles
ParameterManager            = ROOT.ParameterManager
MinuitMinimizationManager   = ROOT.MinuitMinimizationManager
########### PLOTTER / RESULTS RELATED ###########
FitResults                  = ROOT.FitResults
EtaPiPlotGenerator          = ROOT.EtaPiPlotGenerator
PlotGenerator               = ROOT.PlotGenerator
TH1                         = ROOT.TH1
TFile                       = ROOT.TFile
AmplitudeInfo               = ROOT.AmplitudeInfo
gen_amp                     = ROOT.gen_amp

# cmd = 'gen_amp -o test.root -c gen_res.cfg'
# cmd = cmd.split(' ')
generator = ROOT.gen_amp("gen_res.cfg", "test.root")
generator.generate()
