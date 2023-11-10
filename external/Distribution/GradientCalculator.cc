#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include <functional>
#include "MinuitInterface/MinuitParameter.h"
#include "IUAmpTools/report.h"
#include "IUAmpTools/GradientCalculator.h"

const char* GradientCalculator::kModule = "GradientCalculator";

///////////////////////////////////////////////////////////////////////////
// This code is supposed to mirror Minuit's derivative calculation
// as seen in the mnderi function. The source code is located
// https://root.cern/doc/master/TMinuit_8cxx_source.html
// Comments in this file containing L, refer to the physical line number in
// in the TMinuit source code above.
///////////////////////////////////////////////////////////////////////////

using namespace std;

GradientCalculator::GradientCalculator(
        vector< MinuitParameter* >& parameters
    ):  x(parameters){
        npars = parameters.size();
        fGrd.resize(npars, 0);
        fGstep.resize(npars, 0);
        fG2.resize(npars, 0);
        setInitialStepSize(parameters); // should we reset step size every calculate()?
    }

void
GradientCalculator::calculate() {
    double fs1;     // Function value at x + step
    double fs2;     // Function value at x - step
    double step;    // Step size
    double stepb4;  // Previous step size
    double optstp;  // Optimal step size
    double stpmax;  // Maximum step size
    double stpmin;  // Minimum step size
    double grbfor;  // (gr)adient (b)e(for)e
    int fNfcn = 0;  // Number of function calls

    fAmin = FCN(); ++fNfcn; // current value of function

    double dfmin = fEpsma2*8*(abs(fAmin) + fUp); // L2203. dfmin = (d)elta of (f)unction min?
    double tlrstp;  // tlr = tolerance for step size
    double tlrfGrd; // tolerance for gradient
    int ncyc;       // number of cycles for gradient estimation

    // Determine the number of cycles and tolerances based on Strategy
    ncyc = (fIstrat >= 2) ? 5 : (fIstrat == 1) ? 3 : 2;             // L2204 - L2216
    tlrstp = (fIstrat >= 2) ? 0.1 : (fIstrat == 1) ? 0.3 : 0.5;     // L2204 - L2216
    tlrfGrd = (fIstrat >= 2) ? 0.02 : (fIstrat == 1) ? 0.05 : 0.1;  // L2204 - L2216

    // Calculate Gradient
    for (int i = 0; i < npars; ++i) {
        report( DEBUG, kModule ) << "Calculating gradient for parameter " << i << endl;

        double epspri = fEpsma2 + abs(fGrd[i]*fEpsma2); // eps = epsilon? pri = primary?
        double xtf = x[i]->value(); // Original value of x[i]
        stepb4 = 0;

        for (int icyc = 0; icyc < ncyc; ++icyc) {

            // Calculate optimal step size
            optstp = sqrt(dfmin / (abs(fG2[i]) + epspri));
            step = max(optstp, abs(fGstep[i]*0.1));
            if (fGstep[i] < 0 && step > 0.5) step = 0.5;

            // Limit step size
            stpmax = abs(fGstep[i]) * 10;
            step = min(step, stpmax);
            stpmin = abs(fEpsma2 * xtf) * 8;
            step = max(step, stpmin);

            // Check for convergence
            if (abs((step - stepb4) / step) < tlrstp) {
                report( DEBUG, kModule) << " +++ Converged: Relative change in Step size smaller than tolerance" << endl;
                break;
            }

            stepb4 = step;
            fGstep[i] = (fGstep[i] > 0) ? abs(step) : -abs(step);

            // Evaluate function at x + step
            x[i]->setValue( xtf + step );
            fs1 = FCN(); ++fNfcn;

            // Evaluate function at x - step
            x[i]->setValue( xtf - step );
            fs2 = FCN(); ++fNfcn;

            // Calculate first and second derivative
            grbfor = fGrd[i];
            fGrd[i] = (fs1 - fs2) / (2 * step);
            fG2[i] = (fs1 + fs2 - 2 * fAmin) / (step * step);
            x[i]->setValue( xtf ); // Reset x[i] MinuitParameter to the original value

            // Check for convergence
            if (abs(grbfor - fGrd[i]) / (abs(fGrd[i]) + dfmin/step) < tlrfGrd){
                report( DEBUG, kModule) << "(FCN Calls: " << fNfcn << ") --- PAR " << i << " CONVERGED: "
                        " Relative change in gradient smaller than tolerance" << endl;
                break;
            }

            report( DEBUG, kModule) << "(FCN Calls: " << fNfcn << ") +++ PAR " << i << " iterating:  STEP  " << step <<
                    "  GRAD  " << fGrd[i] << "  GRAD2  " << fG2[i] << endl;
        }
    }
    report( DEBUG, kModule) << " +++ Number of function calls: " << fNfcn << endl;
}

void
GradientCalculator::setInitialStepSize(const vector< MinuitParameter* >& parameters) {
    cout << "Setting initial step size" << endl;
    cout << "fEpsma2: " << fEpsma2 << endl;
    cout << "npars: " << parameters.size() << endl;
    for (int i = 0; i < parameters.size(); ++i){
        cout << "fGstep size: " << fGstep.size() << endl;
        cout << " i = " << i << " value: " << parameters[i]->value() << endl;
        cout << abs( parameters[i]->value() )*0.01 + fEpsma2 << endl;
        fGstep[i] = abs( parameters[i]->value() )*0.01 + fEpsma2;
    }
}
