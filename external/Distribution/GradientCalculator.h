#if !defined(GRADIENTCALCULATOR)
#define GRADIENTCALCULATOR

#include <cmath>
#include <vector>
#include <limits>
#include <functional>
#include "MinuitInterface/MinuitParameter.h"

///////////////////////////////////////////////////////////////////////////
// This code is supposed to mirror Minuit's derivative calculation
// as seen in the mnderi function. The source code is located
// https://root.cern/doc/master/TMinuit_8cxx_source.html
// Comments in this file containing L, refer to the physical line number in
// in the TMinuit source code above.
///////////////////////////////////////////////////////////////////////////


using namespace std;

class GradientCalculator {

public:
    GradientCalculator(vector< MinuitParameter* >& parameters);

    ~GradientCalculator() {}

    // Set the Function to evaluate, i.e. likelihood function
    void setFCN( function<double()> fcn ) { FCN = fcn; }

    // Calculate the gradient and FCN value at the current parameter values.
    // 2 FCN calls per parameter per iteration
    //  + 1 FCN call for the initial value
    void calculate();

    // sets Migrad strategy (0=fastest, 1=default, 2=slowest)
    //   See https://iminuit.readthedocs.io/en/stable/faq.html
    void setStrategy(int strategy) { fIstrat = strategy; }

    // Set initial step size
    void setInitialStepSize(const vector< MinuitParameter* >& parameters);

    // Returns the gradients
    const vector<double>& grad_fcn() const { return fGrd; }

    // Returns function value
    double fcn() const { return fAmin; }

private:
    int npars;                          // Number of parameters
    const vector< MinuitParameter* > x; // Current parameter values
    vector<double> fGrd;                // Current gradient
    double fAmin;                       // Current value of function
    vector<double> fGstep;              // Step size for numerical gradient
    vector<double> fG2;                 // Second derivative (approximation)

    // Determine machine precision
    static constexpr double kMachinePrecision = numeric_limits<double>::epsilon(); // L4588 loops to find precision
    double fEpsma2 = sqrt(kMachinePrecision)*2; // L4599 sqrt of machine precision

    double fUp = 0.5; // L4584 for ChiSq fit this is 1. I think this would be 0.5 for Likelihood fits
    function<double()> FCN; // The function to evaluate
    int fIstrat = 1; // Migrad Strategy in TMinuit but here its mainly to set tolerances

    static const char* kModule;
};

#endif // GRADIENTCALCULATOR
