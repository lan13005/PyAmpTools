#!/sur/bin/env python

import os

##########################################################
# This test assumes that cfgfile is initialized
# at the MLE value
##########################################################


def test_gradients():
    PYAMPTOOLS_HOME = os.environ["PYAMPTOOLS_HOME"]
    cfgfile = f"{PYAMPTOOLS_HOME}/tests/samples/SIMPLE_EXAMPLE/fit.cfg"
    assert os.path.isfile(cfgfile), "Config file does not exist at specified path"

    ################### LOAD LIBRARIES ##################
    from pyamptools import atiSetup

    USE_MPI, USE_GPU, RANK_MPI = atiSetup.setup(globals())

    ############## LOAD CONFIGURATION FILE ##############
    parser = ConfigFileParser(cfgfile)
    cfgInfo: ConfigurationInfo = parser.getConfigurationInfo()

    # ############## REGISTER OBJECTS FOR AMPTOOLS ##############
    AmpToolsInterface.registerAmplitude(Zlm())
    AmpToolsInterface.registerAmplitude(BreitWigner())
    AmpToolsInterface.registerDataReader(DataReader())

    ati = AmpToolsInterface(cfgInfo)

    ParameterManager.setDoCovarianceUpdate(False)

    cfgInfo.display()

    nll, gradient = ati.likelihoodAndGradient()
    parMap = dict(ati.parameterManager().getParMap())  # map< string, MinuitParameter* >

    print(f"NLL at the following parameters: {nll}")
    print(f'{"Parameter":<30} {"Value":<30} {"Gradient":<30}')

    for (k, v), g in zip(parMap.items(), gradient):
        v = v.value()
        print(f"{k:<30} {v:<30} {g:<30}")

        assert abs(g) < 1e-3, f"Gradient for parameter {k} at MLE is larger than 1e-3: {g}"
