#if !defined(LassoR)
#define LassoR

#include "IUAmpTools/Neg2LnLikContrib.h"
#include "IUAmpTools/UserNeg2LnLikContrib.h"
#include "GPUManager/GPUCustomTypes.h"

#include <utility>
#include <string>
#include <complex>
#include <vector>
#include <algorithm>

/**
 * Amplitude fits typically include (super)numerous partial waves that
 * can interfere to reproduce what is seen in data. Regularization
 * functions introduces bias into a fit that can help perform
 * waveset selection in a continuous manner
 *
 * LASSO: L1 regularization which introduces a
 * LikelihoodContribution = lambda * sum( |fit fraction| )
 * attempts to zero small components
*/


using std::complex;
using namespace std;

class Lasso : public UserNeg2LnLikContrib< Lasso >{

public:

  Lasso() : UserNeg2LnLikContrib< Lasso >() { }

  Lasso( const vector< string >& args );

  ~Lasso(){ }

  string name() const { return "Lasso"; }

  double neg2LnLikelihood();

private:
  vector< double > m_args;

};

#endif
