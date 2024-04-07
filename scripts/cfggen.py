#!/usr/bin/env python3

import os
import random
from pyamptools import atiSetup


def generate_zlm_cfg(
    EJMs,
    angles,
    fractions,
    datas,
    gens,
    accs,
    bkgnds,
    realAmps,
    fixedAmps,
    fitName,
    cfgFileOutputName,
    basereactName,
    particles,
):
    """
    Generate an AmpTools configuration file for a Zlm fit

    Args:
        EJMs (list): List of lists of [E,J,M] quantum numbers. Reflectivity, spin, spin-projection
        angles (list): List of polarization angles in degrees
        fractions (list): List of polarization fractions
        datas (list): List of data files
        gens (list): List of gen files
        accs (list): List of acc files
        bkgnds (list): List of bkgnd files
        realAmps (list): List of amplitude names that are real
        fixedAmps (list): List of amplitude names that are fixed
        fitName (str): A FitResults (.fit) file will be created with this name prefixed
        cfgFileOutputName (str): Name of output configuration file
        basereactName (str): Base name of reaction
        particles (list): List of particles in reaction

    Returns:
        None, writes a file to cfgFileOutputName
    """
    cfgInfo = ConfigurationInfo(fitName)

    constraintMap = {}
    # reflectivities = set([E for E, J, M in EJMs])
    refTagMap = {1: "p", -1: "m"}  # Positive / Negative
    conjugations = {"Re": "+1", "Im": "-1"}  # Real / Imaginary

    for i in range(len(angles)):
        angle = angles[i]
        fraction = fractions[i]
        data = datas[i]
        gen = gens[i]
        acc = accs[i]
        bkgnd = bkgnds[i]

        reactName = f"{basereactName}_{angle:0>3}"
        scaleName = f"parScale{angle:0>3}"
        parScale = cfgInfo.createParameter(scaleName, 1.0)
        if angle == "0":
            parScale.setFixed(True)

        reactionInfo = cfgInfo.createReaction(reactName, particles)  # ReactionInfo*
        reactionInfo.setData("ROOTDataReader", [data])
        reactionInfo.setGenMC("ROOTDataReader", [gen])
        reactionInfo.setAccMC("ROOTDataReader", [acc])
        reactionInfo.setBkgnd("ROOTDataReader", [bkgnd])

        for conj in conjugations.items():
            conjTag, conjVal = conj

            for ref, J, M in EJMs:
                refTag = refTagMap[ref]

                sumName = f"{refTag}{conjTag}"
                cfgInfo.createCoherentSum(reactName, sumName)  # returns CoherentSumInfo*

                ampName = f"{refTag}{J}{M}"
                ampInfo = cfgInfo.createAmplitude(reactName, sumName, ampName)  # AmplitudeInfo*

                part = "+1" if int(ref) * int(conjVal) > 0 else "-1"
                angularFactor = ["Zlm", f"{J}", f"{M}", conjVal, part, angle, fraction]
                ampInfo.addFactor(angularFactor)
                ampInfo.setScale(f"[{scaleName}]")

                if ampName not in constraintMap:
                    constraintMap[ampName] = [ampInfo]
                else:
                    constraintMap[ampName].append(ampInfo)

    for amp, lines in constraintMap.items():
        value = random.uniform(0.0, 1.0)
        if amp in realAmps:
            lines[0].setReal(True)
        else:
            value += random.uniform(0.0, 1.0) * 1j
        if amp in fixedAmps:
            lines[0].setFixed(True)
        for line in lines[1:]:
            lines[0].addConstraint(line)
        lines[0].setValue(value)
    cfgInfo.display()
    cfgInfo.write(cfgFileOutputName)


if __name__ == "__main__":
    ############## SET ENVIRONMENT VARIABLES ##############
    REPO_HOME = os.environ["REPO_HOME"]

    ################### LOAD LIBRARIES ##################
    atiSetup.setup(globals())

    ############### USER-SPECIFIED VARIABLES ###############
    fitName = "EtaPi"
    cfgFileOutputName = "EtaPi.cfg"
    basereactName = "EtaPi"
    particles = ["Beam", "Proton", "Eta", "Pi0"]

    ### User-specified waveset ###
    EJMs = [
        [1, 2, 2],
        [-1, 2, 2],
        # [ 1,  0,  0],
        # [-1,  0,  0],
    ]
    realAmps = set(["p00", "m00"])
    fixedAmps = []

    ## Load datasets
    baseFolder = f"{REPO_HOME}/tests/samples/REAL_MI_EXAMPLE/samples"

    ## Polarization related, reactNames are scaled by the scales parameters
    ##   Default: parScale0 is fixed to 1.0
    angles = ["0", "45", "90", "135"]
    fractions = ["0.3519", "0.3374", "0.3303", "0.3375"]

    datas = [f"{baseFolder}/data_{angle:0>3}.root" for angle in angles]
    gens = [f"{baseFolder}/gen_{angle:0>3}.root" for angle in angles]
    accs = [f"{baseFolder}/acc_{angle:0>3}.root" for angle in angles]
    bkgnds = [f"{baseFolder}/bkgnd_{angle:0>3}.root" for angle in angles]

    generate_zlm_cfg(
        EJMs,
        angles,
        fractions,
        datas,
        gens,
        accs,
        bkgnds,
        realAmps,
        fixedAmps,
        fitName,
        cfgFileOutputName,
        basereactName,
        particles,
    )
