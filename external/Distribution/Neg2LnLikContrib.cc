#include <map>
#include <string>
#include <cassert>
#include <iostream>
#include <sstream>

#include "IUAmpTools/Neg2LnLikContrib.h"
#include "IUAmpTools/IntensityManager.h"
#include "IUAmpTools/LikelihoodCalculator.h"
#include "IUAmpTools/NormIntInterface.h"

MinuitMinimizationManager* Neg2LnLikContrib::m_minManager;

double Neg2LnLikContrib::operator()(){

  return neg2LnLikelihood();
}

double Neg2LnLikContrib::neg2LnLikelihood(){

  return 0.;
}

string
Neg2LnLikContrib::identifier() const {

  string id = name();

  // make the identifer unique from a likely name in the case of no args
  id += "%%";

  for( vector< string >::const_iterator arg = m_args.begin();
      arg != m_args.end(); ++arg ){

    id += *arg;
    id += " ";
  }

  return id;
}

bool
Neg2LnLikContrib::setParPtr( const string& name, const double* ptr ) const {

  bool foundPar = false;

  for( vector< AmpParameter* >::const_iterator parItr = m_registeredParams.begin();
      parItr != m_registeredParams.end();
      ++parItr ){

    if( (**parItr).name().compare( name ) == 0 ){

      foundPar = true;
      (**parItr).setExternalValue( ptr );

      // pass in the name here to
      // use the const member function here so we only have one const-cast
      // that calls the non-const user function
      updatePar( (**parItr).name() );
    }
  }

  return foundPar;
}

bool
Neg2LnLikContrib::setParValue( const string& name, double val ) const {

  bool foundPar = false;

  for( vector< AmpParameter* >::const_iterator parItr = m_registeredParams.begin();
      parItr != m_registeredParams.end();
      ++parItr ){

    if( (**parItr).name().compare( name ) == 0 ){

      foundPar = true;
      (**parItr).setValue( val );

      // pass in the name here to
      // use the const member function here so we only have one const-cast
      // that calls the non-const user function
      updatePar( (**parItr).name() );
    }
  }

  return foundPar;
}

bool
Neg2LnLikContrib::updatePar( const string& name ) const {

  bool foundPar = false;

  for( vector< AmpParameter* >::const_iterator parItr = m_registeredParams.begin();
      parItr != m_registeredParams.end();
      ++parItr ){

    if( (**parItr).name().compare( name ) == 0 ){

      // The const_cast is a little bit undesirable here.  It can be removed
      // at the expensive of requiring the user to declare all member data in
      // the class that is updated on a parameter update "mutable."
      // Since we are trying to maximize user-friendliness, for now we will
      // remove this potential annoyance.

      const_cast< Neg2LnLikContrib* >(this)->updatePar( **parItr );
      foundPar = true;
    }
  }

  return foundPar;
}

void
Neg2LnLikContrib::registerParameter( AmpParameter& par ){

  m_registeredParams.push_back( &par );
}

void
Neg2LnLikContrib::calcFitFractions( string reactionName, LikelihoodCalculator* likCalc ) {

    // This function mirrors normIntTerm() of LikelihoodCalculator
    // Fit fractions are calculated for each reaction

    bool  m_firstNormIntCalc = likCalc->m_firstNormIntCalc;
    const IntensityManager& m_intenManager = likCalc->intensityManager();
    const NormIntInterface& m_normInt = likCalc->normIntInterface();
    const double* m_normIntArray = m_normInt.normIntMatrix();

    if ( m_prodFactorArrayMap.find(reactionName) == m_prodFactorArrayMap.end() )
      m_prodFactorArrayMap[reactionName] = new double[2*m_intenManager.getTermNames().size()];
    double* m_prodFactorArray = m_prodFactorArrayMap[reactionName];

    // check to be sure we can actually perform a computation of the
    // normalization integrals in case we have floating parameters

    if( m_intenManager.hasTermWithFreeParam() && !m_normInt.hasAccessToMC() ){
      assert( false );
    }

    if( ( m_firstNormIntCalc && m_normInt.hasAccessToMC() ) ||
        ( m_intenManager.hasTermWithFreeParam() && !m_firstNormIntCalc ) ){

      m_normInt.forceCacheUpdate( true );
    }

    termNames = m_intenManager.getTermNames();
    int n = termNames.size();
    m_intenManager.prodFactorArray( m_prodFactorArray );

    double inputTotal = 0; // Normalization for fit_fraction
    fit_fractions.clear(); // Clear fit fractions from previous iteration

    switch( m_intenManager.type() ){

      case IntensityManager::kAmplitude:

        for( int a = 0; a < n; ++a ){
          for( int b = 0; b <= a; ++b ){

            double thisTerm = 0;

            double reVa = m_prodFactorArray[ 2*a   ];
            double imVa = m_prodFactorArray[ 2*a+1 ];
            double reVb = m_prodFactorArray[ 2*b   ];
            double imVb = m_prodFactorArray[ 2*b+1 ];
            double reNI = m_normIntArray[ 2*a*n+2*b   ];
            double imNI = m_normIntArray[ 2*a*n+2*b+1 ];

            thisTerm  = ( reVa*reVb + imVa*imVb ) * reNI;
            thisTerm -= ( imVa*reVb - reVa*imVb ) * imNI;

            if( a != b ) thisTerm *= 2;

            inputTotal += thisTerm;

            if( a == b ) fit_fractions.push_back(thisTerm);
          }
        }
        break;

      case IntensityManager::kMoment:

        break;

      default:
        assert( false );
        break;
    } // end switch

    // Update underlying LikelihoodCalculator
    likCalc->m_firstNormIntCalc = false;

    // Determine the fit fractions
    for (unsigned int i = 0; i < fit_fractions.size(); i++){
      fit_fractions[i] /= inputTotal;
    }
}
