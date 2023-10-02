from utils import AmplitudeParameters
import argparse

parser = argparse.ArgumentParser(description='Test AmplitudeParameters class')
parser.add_argument('fitName', type=str, help='Name of fit result file')

REPO_HOME = os.environ['REPO_HOME']
fitName = f'{REPO_HOME}/gen_amp/result.fit'
os.environ['ATI_USE_MPI'] = "0" # set to 1 to use MPI libraries
os.environ['ATI_USE_GPU'] = "0"
from atiSetup import *

results = FitResults( fitName )
if not results.valid():
    print(f'Invalid fit result in file: {fitName}')
    exit()

amplitudeParameters = AmplitudeParameters()
amplitudeParameters.load_cfg(results)
parameters = amplitudeParameters.flatten_parameters()
unflattened = amplitudeParameters.unflatten_parameters(parameters)
assert ( unflattened == amplitudeParameters.uniqueAmps ), "Unflattened parameters do not match original parameters"
uniqueAmps = {'etapi::imZ::resAmp1': (1116.45515738103+0j), 'etapi::imZ::resAmp2': (0.99054260126944-5.97630945097717j), 'etapi::imZ::resAmp3': (-1.99353200931724-33.9805078068182j)}
uniqueReal = {'etapi::imZ::resAmp1': True, 'etapi::imZ::resAmp2': False, 'etapi::imZ::resAmp3': False}
parameters = amplitudeParameters.flatten_parameters(uniqueAmps, uniqueReal)
unflattened = amplitudeParameters.unflatten_parameters(parameters, uniqueReal)
assert ( unflattened == amplitudeParameters.uniqueAmps ), "Unflattened parameters do not match original parameters"
