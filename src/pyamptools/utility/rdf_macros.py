import os

import ROOT

#############################################
# Load FSRoot's FSMath into ROOT interpreter to be used in ROOT's RDataFrames
# Does not look easy to use the macros directly with RDF explicitly
#   here we will just mimic their notation and write inspired code
# SOURCES: FSBasic/FSMath.C
#          FSBasic/FSTree.C
#############################################


def loadMacros():
    """
    Load Macros that mirror FSRoot's FSMath
    """

    # Load /home/lng/WORK/WORK/PyAmpTools9/utility/RDF_userDefFuncs.cc
    #   which contains the FSMath functions
    if "PYAMPTOOLS_HOME" not in os.environ:
        print("PYAMPTOOLS_HOME not set. Cannot load FSMath macros.")
        return
    else:
        PYAMPTOOLS_HOME = os.getenv("PYAMPTOOLS_HOME")
    print(f"Loading FSMath macros from {PYAMPTOOLS_HOME}/utility/RDF_userDefFuncs.cc")
    ROOT.gROOT.LoadMacro(f"{PYAMPTOOLS_HOME}/utility/RDF_userDefFuncs.cc")

    ROOT.gInterpreter.Declare("""
    #include "FSBasic/FSMath.h"
    using Vec_t = const ROOT::RVec<float>&;

    // PYAMPTOOLS DEFINED - see RDF_userDefFuncs.cc
    double UNWRAP( double phi ){ return PyAmpToolsMath::unwrap(phi); }
    double BIGPHI( float polAngle, Vec_t P1, Vec_t P2, Vec_t P3, Vec_t P4 ){return PyAmpToolsMath::bigPhi( polAngle, P1[0], P1[1], P1[2], P1[3], P2[0], P2[1], P2[2], P2[3], P3[0], P3[1], P3[2], P3[3], P4[0], P4[1], P4[2], P4[3]); }

    // ANGULAR QUANTITIES
    double HELPHI( Vec_t P1, Vec_t P2, Vec_t P3, Vec_t P4 ){return FSMath::helphi( P1[0], P1[1], P1[2], P1[3], P2[0], P2[1], P2[2], P2[3], P3[0], P3[1], P3[2], P3[3], P4[0], P4[1], P4[2], P4[3]); }
    double GJPHI(  Vec_t P1, Vec_t P2, Vec_t P3, Vec_t P4 ){return FSMath::gjphi(  P1[0], P1[1], P1[2], P1[3], P2[0], P2[1], P2[2], P2[3], P3[0], P3[1], P3[2], P3[3], P4[0], P4[1], P4[2], P4[3]); }
    double PRODCOSTHETA( Vec_t P1, Vec_t P2, Vec_t P3 ){return FSMath::prodcostheta( P1[0], P1[1], P1[2], P1[3], P2[0], P2[1], P2[2], P2[3], P3[0], P3[1], P3[2], P3[3] ); }
    double PLANEPHI(     Vec_t P1, Vec_t P2, Vec_t P3 ){return FSMath::planephi(     P1[0], P1[1], P1[2], P1[3], P2[0], P2[1], P2[2], P2[3], P3[0], P3[1], P3[2], P3[3] ); }
    double ENERGY( Vec_t P1, Vec_t P2){return FSMath::boostEnergy( P1[0], P1[1], P1[2], P1[3], P2[0], P2[1], P2[2], P2[3]); }
    //   3 Particle versions of costheta
    double HELCOSTHETA( Vec_t P1, Vec_t P2, Vec_t P3 ){return FSMath::helcostheta( P1[0], P1[1], P1[2], P1[3], P2[0], P2[1], P2[2], P2[3], P3[0], P3[1], P3[2], P3[3] ); }
    double GJCOSTHETA(  Vec_t P1, Vec_t P2, Vec_t P3 ){return FSMath::gjcostheta(  P1[0], P1[1], P1[2], P1[3], P2[0], P2[1], P2[2], P2[3], P3[0], P3[1], P3[2], P3[3] ); }
    //   4 Particle versions of costheta
    double HELCOSTHETA( Vec_t P1, Vec_t P2, Vec_t P3, Vec_t P4 ){return FSMath::helcostheta( P1[0], P1[1], P1[2], P1[3], P2[0], P2[1], P2[2], P2[3], P3[0], P3[1], P3[2], P3[3] ); }
    double GJCOSTHETA(  Vec_t P1, Vec_t P2, Vec_t P3, Vec_t P4 ){return FSMath::gjcostheta(  P1[0], P1[1], P1[2], P1[3], P2[0], P2[1], P2[2], P2[3], P3[0], P3[1], P3[2], P3[3] ); }

    // MASS
    double MASS(Vec_t P1, Vec_t P2, Vec_t P3 ){ return sqrt(pow(P1[3]+P2[3]+P3[3],2) - pow(P1[0]+P2[0]+P3[0],2) - pow(P1[1]+P2[1]+P3[1],2) - pow(P1[2]+P2[2]+P3[2],2)); }
    double MASS(Vec_t P1, Vec_t P2           ){ return sqrt(pow(P1[3]+P2[3],2)       - pow(P1[0]+P2[0],2)       - pow(P1[1]+P2[1],2)       - pow(P1[2]+P2[2],2)); }
    double MASS(Vec_t P1                     ){ return sqrt(pow(P1[3],2)             - pow(P1[0],2)             - pow(P1[1],2)             - pow(P1[2],2)); }
    double MASS2(Vec_t P1, Vec_t P2, Vec_t P3){ return     (pow(P1[3]+P2[3]+P3[3],2) - pow(P1[0]+P2[0]+P3[0],2) - pow(P1[1]+P2[1]+P3[1],2) - pow(P1[2]+P2[2]+P3[2],2)); }
    double MASS2(Vec_t P1, Vec_t P2          ){ return     (pow(P1[3]+P2[3],2)       - pow(P1[0]+P2[0],2)       - pow(P1[1]+P2[1],2)       - pow(P1[2]+P2[2],2)); }
    double MASS2(Vec_t P1                    ){ return     (pow(P1[3],2)             - pow(P1[0],2)             - pow(P1[1],2)             - pow(P1[2],2)); }

    // MANDELSTAM
    double T(Vec_t P1, Vec_t P2){ return -1 * ( pow(P1[3] - P2[3],2) - pow(P1[0] - P2[0],2) - pow(P1[1] - P2[1],2) - pow(P1[2] - P2[2],2) ); }

    // MOMENTUM COMPONENTS
    double MOMENTUMX(Vec_t P1){ return P1[0]; }
    double MOMENTUMY(Vec_t P1){ return P1[1]; }
    double MOMENTUMZ(Vec_t P1){ return P1[2]; }
    double ENERGY(   Vec_t P1){ return P1[3]; }
    double MOMENTUMR(Vec_t P1){ return sqrt(pow(P1[0],2) + pow(P1[1],2)               ); }
    double MOMENTUM( Vec_t P1){ return sqrt(pow(P1[0],2) + pow(P1[1],2) + pow(P1[2],2)); }

    // MATH
    double DOTPRODUCT(Vec_t P1, Vec_t P2){ return P1[0]*P2[0] + P1[1]*P2[1] + P1[2]*P2[2]; }
    double COSINE(Vec_t P1){ return P1[2]/MOMENTUM(P1); }
    double COSINE(Vec_t P1, Vec_t P2){ return DOTPRODUCT(P1, P2)/(MOMENTUM(P1)*MOMENTUM(P2)); }
    """)
