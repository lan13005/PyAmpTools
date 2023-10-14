import os
from amploader import AmplitudeParameters
from atiSetup import *
import pytest

@pytest.mark.amppars
def test_AmpPars():
    REPO_HOME = os.environ["REPO_HOME"]
    fitName = f'{REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/result.fit'

    results = FitResults( fitName )
    if not results.valid():
        print(f'Invalid fit result in file: {fitName}')
        exit()

    amplitudeParameters = AmplitudeParameters()
    amplitudeParameters.load_cfg(results)
    parameters = amplitudeParameters.flatten_parameters()
    unflattened = amplitudeParameters.unflatten_parameters(parameters)
    print(f'Unflattened parameters: {unflattened}')
    print(f'Unique amplitudes: {amplitudeParameters.uniqueAmps}')
    assert ( unflattened == amplitudeParameters.uniqueAmps ), "Unflattened parameters do not match original parameters"
    uniqueAmps = {'etapi::imZ::resAmp1': (120.707457568065+0j), 'etapi::imZ::resAmp2': (144.725598248866-43.1406410313654j), 'etapi::imZ::resAmp3': (234.391084722308+0j)}
    uniqueReal = {'etapi::imZ::resAmp1': True, 'etapi::imZ::resAmp2': False, 'etapi::imZ::resAmp3': False}
    parameters = amplitudeParameters.flatten_parameters(uniqueAmps, uniqueReal)
    unflattened = amplitudeParameters.unflatten_parameters(parameters, uniqueReal)
    print(f'Unflattened parameters: {unflattened}')
    print(f'Unique amplitudes: {amplitudeParameters.uniqueAmps}')
    assert ( unflattened == amplitudeParameters.uniqueAmps ), "Unflattened parameters do not match original parameters"
