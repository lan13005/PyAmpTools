import ROOT
import os
from typing import List
from atiSetup import *

def runTest():
    ############## SET ENVIRONMENT VARIABLES ##############
    REPO_HOME     = os.environ['REPO_HOME']

    ############## LOAD CONFIGURATION FILE ##############
    cfgfile = f'{REPO_HOME}/tests/samples/fit_res.cfg'
    parser = ConfigFileParser(cfgfile)
    cfgInfo: ConfigurationInfo = parser.getConfigurationInfo()
    cfgInfo.display()

    ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude( Zlm() )
    AmpToolsInterface.registerDataReader( ROOTDataReader() )

    ati = AmpToolsInterface( cfgInfo )
    parMgr: ParameterManager = ati.parameterManager()

    key = 'etapi::imZ::resAmp1'
    nProdPars = len(parMgr)
    prefit_nll = ati.likelihood()
    parMgr[key] = complex(15,10)
    par_real, par_imag = parMgr[key].real, parMgr[key].imag
    post_nll = ati.likelihood()

    return nProdPars, prefit_nll, par_real, par_imag, post_nll
