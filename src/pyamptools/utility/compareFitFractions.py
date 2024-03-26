from pyamptools import atiSetup
import argparse

########################################
# Python port of AmpTools/AmpTools/ROOT/compareFitFractions.C
########################################

# The goal of this function is to facilitate easier input/output
# checks where one generates MC and then does a fit to check
# the fit result is consistent with generation.  The general problem
# is that the production coefficiencts at generation time have
# a non-trivial relationship with those that come from the fit.
# The easiest way to compare is to compare fit fractions, but
# computing the fit fractions is not possible without the
# normalization integrals, which are usually not computed as
# part of the generation.  This script **ASSUMES** the normalization
# integrals in the .fit output are true integrals of the amplitudes
# configured in the generator.  (This will NOT be true if for example,
# amplitudes have embedded parameters that are floating in the fit
# and they float to values different than those initialized in the
# generation.)  As long as this assumption is valid, then the generated
# fit fractions can be computed with the post-fit integrals and the
# generated production parameters in the config file used for generation.
# The computation of fit fractions from the fit uses standard functions.
# Uncertainty computations and therefore pulls are likely not
# rigorous and should be used as rough indicators of problems.


def compareFitFractions(
        genCfgFile: str,
        fitFile: str,
        verbose: bool = True,
):
  ############## SET ENVIRONMENT VARIABLES ##############
  atiSetup.setup(globals())

  fr = FitResults( fitFile )

  cfParser = ConfigFileParser( genCfgFile )
  cfgInfo = cfParser.getConfigurationInfo()

  reactions = cfgInfo.reactionList() # vector< ReactionInfo* >

  for reacInfo in reactions:
    reaction = reacInfo.reactionName() # string

    ampList = cfgInfo.amplitudeList( reaction ) # vector< AmplitudeInfo* >
    ni = fr.normInt( reaction ) # const NormIntInterface*

    # This python port will take a roundabout way to access the scale factors
    # Here, we create a dictionary of the parameter names: values.
    # parameters have the form [parName] in the config file
    parMap = {}
    for parInfo in cfgInfo.parameterList():
      parMap[ f'[{parInfo.parName()}]' ] = parInfo.value()

    #  ** first nested loop over the generated production parameters and
    #     the results of the normalization integrals to pick out all
    #     of the relevant info needed to construct the generated fit
    #     fractions

    # need to renormalize the input PDF based using the input parameters
    # and the integrals of the PDF from the fit file
    inputTotal = complex( 0, 0 )

    nameGenFrac = {} # map< string, double >
    ampNameVec = [] # vector< string >

    for ampInfo in ampList:

      ampName = ampInfo.fullName() # string
      ampNameVec.append( ampName )

      inputProdAmp = parMap[ ampInfo.scale() ] * ampInfo.value() # complex< double >

      # this is the unnormalized generated intensity
      nameGenFrac[ampName] = abs( inputProdAmp * inputProdAmp.conjugate() ) \
                                  * abs( ni.ampInt( ampName, ampName ) )

      # need to compute the "generated total" in order to renormalize
      # the generated production parameters
      for conjAmpInfo in ampList:

        inputConjProdAmp = parMap[ conjAmpInfo.scale() ] * conjAmpInfo.value()

        inputTotal += inputProdAmp * inputConjProdAmp.conjugate() \
                                   * ni.ampInt( ampName, conjAmpInfo.fullName() )

    # do a check to see if this looks like a real number
    if inputTotal.imag  > 1E-6:
      print(f'WARNING:  normalization factor should be real: {inputTotal}')

    # ** second loop over the amplitude names -- renormalize to produce
    #    the fit fractions and print things to the screen

    fitTotal = fr.intensity( ampNameVec ) # pair< double, double >


    if verbose:
      print("\n****************************************************************************")
      print(f"REACTION:  {reaction}\n")
      print(f"{'Amplitude':<35} {'Generated Fraction [%]':<30} {'Fit Result Fraction [%]':<30}  Pull")
      print(f"{'---------':<35} {'----------------------':<30} {'-----------------------':<30}  ----")

    results = {
      'amplitude': [],
      'generated': [],
      'fitted': [],
      'fitted_err': [],
      'pull': [],
    }
    for name in ampNameVec:

      # now record the fit fractions
      fitFrac = fr.intensity( [name] ) # pair<double,double>
      fitFrac.first /= fitTotal.first * 0.01
      fitFrac.second /= fitTotal.first * 0.01
      nameGenFrac[name] /= abs( inputTotal ) * 0.01

      generated = nameGenFrac[name]
      fitted = fitFrac.first
      fitted_err = fitFrac.second
      pull = (fitted - generated) / fitted_err

      results['amplitude'].append(name)
      results['generated'].append(generated)
      results['fitted'].append(fitted)
      results['fitted_err'].append(fitted_err)
      results['pull'].append(pull)

      if verbose:
        print(f'{name:<35} {generated:<30.3f} {fitted:<5.2f} +/- {fitted_err:<9.2f} \
              {(fitted - generated) / fitted_err:<4.2f}')

    return results

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Compare fit fractions')
  parser.add_argument('genCfgFile', help='Generator config file')
  parser.add_argument('fitFile',    help='Fit file')
  args = parser.parse_args()

  compareFitFractions( args.genCfgFile, args.fitFile )
