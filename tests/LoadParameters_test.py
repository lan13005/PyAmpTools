import ROOT
import os
import atiSetup
import pytest
from LoadParameters import LoadParameters
from utils import testKnownFailure

############## SET ENVIRONMENT VARIABLES ##############
REPO_HOME     = os.environ['REPO_HOME']
atiSetup.setup(globals())

def performCheck(input, truths):
    true_params, true_paramsIsReal, true_flat_params, true_names = truths

    paramCtrl = LoadParameters(input)

    cfgInfo = paramCtrl.cfg
    # cfgInfo.display()

    ############# TEST LOADING OF PARAMETERS #############
    print(f'paramCtrl.params = {paramCtrl.params}')
    print(f'true_params = {true_params}')
    assert( paramCtrl.params ==  true_params )
    assert( paramCtrl.paramsIsReal == true_paramsIsReal )

    ############# TEST FLATTENING OF PARAMETERS #############
    flat_params, keys, names = paramCtrl.flatten_parameters()
    assert( flat_params == true_flat_params )
    assert( names == true_names )

    ############# TEST KNOWN FAILURES #############
    testKnownFailure(paramCtrl.flatten_parameters)({'width1':1000}) # width1 is fixed
    # etapi::reZ::resAmp1 constrained to imZ and imZ was parsed first
    testKnownFailure(paramCtrl.flatten_parameters)({'etapi::reZ::resAmp1':1000})

    ############ TEST REAL PRODUCTION PARAMETERS ########
    flat_params, keys, names = paramCtrl.flatten_parameters(
        {'mass1': 1.2, 'etapi::imZ::resAmp1' : 100-200j, }
    )
    assert( flat_params == [1.2, 100.0] ) # etapi::imZ::resAmp1 was set as real

    flat_params, keys, names = paramCtrl.flatten_parameters(
        {'mass1': 1.2, 'etapi::imZ::resAmp2' : 100-200j, }
    )
    assert( flat_params == [1.2, 100.0, -200.0] ) # etapi::imZ::resAmp1 was set as real


@pytest.mark.loadparameters
def test_cfg_LoadParameters():
    print('\nTesting config input:\n===========================')
    file = f'{REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/fit_fakeParams.cfg'
    parser = ConfigFileParser(file)
    input_cfg = parser.getConfigurationInfo()
    true_params = {'etapi::imZ::resAmp1': (10+0j), 'etapi::imZ::resAmp2': (10+0j),
                   'etapi::imZ::resAmp3': (10+0j), 'mass1': 1.0, 'mass2': 2.0,
                   'width2': 200.0, 'width3': 300.0}
    true_paramsIsReal = {'etapi::imZ::resAmp1': True, 'etapi::imZ::resAmp2': False,
                         'etapi::imZ::resAmp3': False, 'mass1': True, 'mass2': True,
                         'width2': True, 'width3': True}
    true_flat_params = [10.0, 10.0, 0.0, 10.0, 0.0, 1.0, 2.0, 200.0, 300.0]
    true_names = ['Re[resAmp1]', 'Re[resAmp2]', 'Im[resAmp2]', 'Re[resAmp3]', 'Im[resAmp3]',
                  'mass1', 'mass2', 'width2', 'width3']
    truths = [true_params, true_paramsIsReal, true_flat_params, true_names]
    performCheck(input_cfg, truths)

@pytest.mark.loadparameters
def test_result_LoadParameters():
    print('\nTesting fit result input:\n===========================')
    file = f'{REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/result_fakeParams.fit'
    results = FitResults( file )
    true_params = {'etapi::imZ::resAmp1': (462.405000586143+0j),
                   'etapi::imZ::resAmp2': (279.819030071884-16.633970687601j),
                   'etapi::imZ::resAmp3': (406.191515689351+164.391119358749j),
                   'mass1': 2.38827351798476, 'mass2': 2.94879810784617, 'width2': 5.30501566756126, 'width3': 0.0}
    true_paramsIsReal = {'etapi::imZ::resAmp1': True, 'etapi::imZ::resAmp2': False, 'etapi::imZ::resAmp3':
                         False, 'mass1': True, 'mass2': True, 'width2': True, 'width3': True}
    true_flat_params = [462.405000586143, 279.819030071884, -16.633970687601, 406.191515689351,
                        164.391119358749, 2.38827351798476, 2.94879810784617, 5.30501566756126, 0.0]
    true_names = ['Re[resAmp1]', 'Re[resAmp2]', 'Im[resAmp2]', 'Re[resAmp3]', 'Im[resAmp3]', 'mass1',
                  'mass2', 'width2', 'width3']
    truths = [true_params, true_paramsIsReal, true_flat_params, true_names]
    performCheck(results, truths)
