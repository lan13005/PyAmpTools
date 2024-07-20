#include "TLorentzVector.h"
#include "TVector3.h"
#include "TMath.h"

///////////////////////////////////////
// RDF user defined functions to be used in rdf_macros.py
///////////////////////////////////////

class PyAmpToolsMath {
    public:
	    // for beam + target -> X + recoil and X -> a + b
	    //      P               X     R        X    A   B
	    // azimuthal angle of photon polarization vector in R rest frame [rad]
	    // code taken from https://github.com/JeffersonLab/halld_sim/blob/538677ee1347891ccefa5780e01b158e035b49b1/src/libraries/AMPTOOLS_AMPS/TwoPiAngles.cc#L94
	    static double
	    bigPhi(
			const double polAngle,
			const double PxP, const double PyP, const double PzP, const double EnP,
			const double PxR, const double PyR, const double PzR, const double EnR,
			const double PxA, const double PyA, const double PzA, const double EnA,
			const double PxB, const double PyB, const double PzB, const double EnB
	    ) {
			TLorentzVector beam   ( PxP, PyP, PzP, EnP );
			TLorentzVector recoil ( PxR, PyR, PzR, EnR );
			TLorentzVector p1     ( PxA, PyA, PzA, EnA ); 
			TLorentzVector p2     ( PxB, PyB, PzB, EnB );
			TLorentzVector target(0,0,0,0.9382719);

			TLorentzVector com = beam + target;
			TLorentzRotation comBoost( -com.BoostVector() );
			TLorentzVector beam_com = comBoost * beam;
			TLorentzVector recoil_com = comBoost * recoil;
			TLorentzVector p1_com = comBoost * p1;
			TLorentzVector p2_com = comBoost * p2;
			TLorentzVector resonance_com = p1_com + p2_com;

			TLorentzRotation resonanceBoost( -resonance_com.BoostVector() );
				
			TLorentzVector beam_res = resonanceBoost * beam_com;
			TLorentzVector recoil_res = resonanceBoost * recoil_com;
			TLorentzVector p1_res = resonanceBoost * p1_com;
				
			// normal to the production plane
			TVector3 y = (beam.Vect().Unit().Cross(-recoil.Vect().Unit())).Unit();

			// choose helicity frame: z-axis opposite recoil proton in rho rest frame
			TVector3 z = -1. * recoil_res.Vect().Unit();
			TVector3 x = y.Cross(z).Unit();
			TVector3 angles( (p1_res.Vect()).Dot(x),
					(p1_res.Vect()).Dot(y),
					(p1_res.Vect()).Dot(z) );

			// float deg2rad = 0.01745;
			TVector3 eps(TMath::Cos(polAngle*0.01745), TMath::Sin(polAngle*0.01745), 0.0); // beam polarization vector
			return atan2(y.Dot(eps), beam.Vect().Unit().Dot(eps.Cross(y)));
	    }

		static double
		unwrap(double phi) {
			while (phi < -TMath::Pi()) phi += 2*TMath::Pi();
			while (phi > TMath::Pi()) phi -= 2*TMath::Pi();
			return phi;
		}
};
