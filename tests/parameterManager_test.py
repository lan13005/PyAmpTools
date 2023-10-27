import ROOT
import os
from typing import List
from atiSetup import atiSetup
import pytest

@pytest.mark.parmgr
def test_parMgr():
    ############## SET ENVIRONMENT VARIABLES ##############
    REPO_HOME     = os.environ['REPO_HOME']
    atiSetup(globals())

    ############## LOAD CONFIGURATION FILE ##############
    cfgfile = f'{REPO_HOME}/tests/samples/SIMPLE_EXAMPLE/fit.cfg'
    parser = ConfigFileParser(cfgfile)
    cfgInfo: ConfigurationInfo = parser.getConfigurationInfo()
    cfgInfo.display()

    ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude( Zlm() )
    AmpToolsInterface.registerDataReader( DataReader() )

    ati = AmpToolsInterface( cfgInfo )
    parMgr: ParameterManager = ati.parameterManager()

    key = 'etapi::imZ::resAmp1'
    nProdPars = len(parMgr)
    prefit_nll = ati.likelihood()
    parMgr[key] = complex(15,10)
    par_real, par_imag = parMgr[key].real, parMgr[key].imag
    post_nll = ati.likelihood()

    assert( nProdPars == 6 )
    assert( prefit_nll != 1e6 and prefit_nll is not None )
    assert( par_real == 15 and par_imag == 0 )
    assert( post_nll != 1e6 and post_nll is not None )
