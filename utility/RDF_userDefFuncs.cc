#include "TLorentzVector.h"
#include "TVector3.h"
#include "TMath.h"

///////////////////////////////////////
// RDF user defined functions to be used in RDFMacros.py
///////////////////////////////////////

class RDF_CFuncs {
    public:
	    // for beam + target -> X + recoil and X -> a + b
	    //      D               R     C        R    A   B
	    // azimuthal angle of photon polarization vector in R rest frame [rad]
	    // BIGPHI(a; b; recoil; beam)
	    // code taken from (B. Grube) from https://github.com/JeffersonLab/halld_sim/blob/538677ee1347891ccefa5780e01b158e035b49b1/src/libraries/AMPTOOLS_AMPS/TwoPiAngles.cc#L94
	    static double
	    bigPhi(
	        const double PxPC, const double PyPC, const double PzPC, const double EnPC,
	        const double PxPD, const double PyPD, const double PzPD, const double EnPD,
	        const double polAngle  // [deg]
	    ) {
	        TLorentzVector recoil(PxPC, PyPC, PzPC, EnPC);
	        TLorentzVector beam  (PxPD, PyPD, PzPD, EnPD);
	        const TVector3 yAxis = (beam.Vect().Unit().Cross(-recoil.Vect().Unit())).Unit();  // normal of production plane in lab frame
	        const TVector3 eps(1, 0, 0);  // reference beam polarization vector at 0 degrees in lab frame
	        double Phi = polAngle * TMath::DegToRad() + atan2(yAxis.Dot(eps), beam.Vect().Unit().Dot(eps.Cross(yAxis)));  // angle in lab frame [rad]
	        // ensure [-pi, +pi] range
	        while (Phi > TMath::Pi()) {
	            Phi -= TMath::TwoPi();
	        }
	        while (Phi < -TMath::Pi()) {
	            Phi += TMath::TwoPi();
	        }
	        assert(polAngle == 45);
	        return Phi;
	    }
}
