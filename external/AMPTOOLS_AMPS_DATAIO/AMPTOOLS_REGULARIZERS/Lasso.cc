#include <cassert>
#include <iostream>
#include <string>
#include <complex>
#include <cstdlib>
#include <fstream>

#include "AMPTOOLS_REGULARIZERS/Lasso.h"
#include "IUAmpTools/LikelihoodCalculator.h"

Lasso::Lasso( const vector< string >& args ) :
UserNeg2LnLikContrib< Lasso >(args)
{
  for( unsigned int i = 0; i < args.size(); ++i ){
    m_args.push_back( atof( args[i].c_str() ) );
    cout << "  Argument " << i << ": " << m_args[i] << endl;
  }
}

double
Lasso::neg2LnLikelihood(){

  // Now we can calculate the regularization
  // NLL ~ lambda * sum( |fit fraction| )
  // Each reaction has its own LikelihoodCalculator
  // Loop over all of them and add up their contribution

  double regularization=0;
  // m_likCalcMap is a map reactionName : LikelihoodCalculator*
  for (auto const& x : m_likCalcMap){
    calcFitFractions( x.first, x.second );
    for (unsigned int i = 0; i < fit_fractions.size(); i++){
      cout << "Fit Fraction of " << termNames[i]
                               << ": " << fit_fractions[i] << endl;
      regularization += m_args[0] * abs( fit_fractions[i] );
    }
  }
  cout << "Regularization: " << regularization << endl;
  return regularization;
}
