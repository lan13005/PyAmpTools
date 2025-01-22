import os
from pyamptools import atiSetup


def test_parMgr():
    ############## SET ENVIRONMENT VARIABLES ##############
    PYAMPTOOLS_HOME = os.environ["PYAMPTOOLS_HOME"]
    atiSetup.setup(globals())

    ############## LOAD CONFIGURATION FILE ##############
    cfgfile = f"{PYAMPTOOLS_HOME}/tests/samples/SIMPLE_EXAMPLE/fit.cfg"
    parser = ConfigFileParser(cfgfile)
    cfgInfo: ConfigurationInfo = parser.getConfigurationInfo()
    cfgInfo.display()

    ati = AmpToolsInterface(cfgInfo)
    parMgr: ParameterManager = ati.parameterManager()

    print("Printing free parameters:")
    parMap = dict(parMgr.getParMap())
    for k, v in parMap.items():
        print(f"{k}: {v.value()}")
    print()

    key = "etapi::imZ::resAmp1_re"
    assert key in parMgr

    nProdPars = len(parMgr)
    assert nProdPars == 5

    prefit_nll = ati.likelihood()
    print("Prefit NLL:", prefit_nll)
    assert prefit_nll == 14346.408126566828 and prefit_nll is not None

    # we are manually parameters, no available covariance from Minuit
    parMgr.setDoCovarianceUpdate(False)

    parMgr[key] = 15.0
    assert parMgr[key] == 15.0

    post_nll = ati.likelihood()
    print("Postfit NLL:", post_nll)
    assert post_nll == 23991.13274997979 and post_nll is not None
