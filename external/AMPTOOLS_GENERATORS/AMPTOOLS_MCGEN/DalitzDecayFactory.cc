
#include <vector>
#include <assert.h>
#include <stdlib.h>

#include "AMPTOOLS_MCGEN/DalitzDecayFactory.h"

#include "TLorentzVector.h"
#include "TLorentzRotation.h"

const double DalitzDecayFactory::kPi = 3.14159;

DalitzDecayFactory::DalitzDecayFactory( double parentMass, const vector<double>& childMass ) :
m_parentMass( parentMass ),
m_childMass( childMass )
{

	// for the decay X -> 0 1 2, the max momentum of 2 in X RF is
	// m_X / 2, and max momentum of 0 in 01 RF is near m_X / 2 also

	m_maxLorentzFactor = 0.25 * m_parentMass * m_parentMass;

	assert( childMass.size() == 3 );
}


vector<TLorentzVector>
DalitzDecayFactory::generateDecay() const {

	vector<TLorentzVector> child( 3 );
	vector<TVector3> childMom( 3 );

	// create some useful temporary variables
	TLorentzVector isobar;
	TVector3 isobarMom;
	double isobarMass;

	do{

		// generate the mass of the isobar ( 01 --> 0 1 )
		isobarMass = random( ( m_childMass[0] + m_childMass[1] ),
							 ( m_parentMass - m_childMass[2] ) );

		// let the X decay to isobar + 2 in the X CM
		// fill the isobar momentum vector and the bachelor momentum vector
		isobarMom.
			SetMagThetaPhi( cmMomentum( m_parentMass, isobarMass, m_childMass[2] ),
						  acos( random( -0.999999, 0.999999 ) ),
						  random( -kPi, kPi ) );
		childMom[2] = -isobarMom;

		// setup the isobar 4 vector
		isobar.SetVect( isobarMom );
		isobar.SetE( sqrt( isobarMom.Mag2() + isobarMass * isobarMass ) );

		// let the isobar decay to 0 1 in the isobar CM
		childMom[0].
			SetMagThetaPhi( cmMomentum( isobarMass, m_childMass[0], m_childMass[1] ),
						  acos( random( -0.999999, 0.999999 ) ),
						  random( -kPi, kPi ) );
		childMom[1] = -childMom[0];

		// now we have childMom[0] and childMom[1] in the isobar rest frame
		// and childMom[2] in the resonance rest frame
	}
	while( ( childMom[0].Mag() * isobarMom.Mag() ) <
		   random( 0.0, m_maxLorentzFactor ) );

	// fill the final four-vectors
	for( int i = 0; i < 3; ++i ){

		child[i].SetVect( childMom[i] );
		child[i].SetE( sqrt( childMom[i].Mag2() +
							 m_childMass[i] * m_childMass[i] ) );
	}

	// boost the isobar children to the resonance rest frame
	child[0].Boost( isobar.BoostVector() );
	child[1].Boost( isobar.BoostVector() );

	return child;
}

double
DalitzDecayFactory::cmMomentum( double M, double m1, double m2 ) const {

	// mini PDG Eq: 38.16

	double num1 = ( M * M - ( m1 + m2 ) * ( m1 + m2 ) );
	double num2 = ( M * M - ( m1 - m2 ) * ( m1 - m2 ) );

	return( sqrt( num1 * num2 ) / ( 2 * M ) );
}

double
DalitzDecayFactory::random( double low, double hi ) const {

	return( ( hi - low ) * drand48() + low );
}
