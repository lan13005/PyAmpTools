#if !defined(AMPTOOLSINTERFACE)
#define AMPTOOLSINTERFACE

//******************************************************************************
// This file is part of AmpTools, a package for performing Amplitude Analysis
//
// Copyright Trustees of Indiana University 2010, all rights reserved
//
// This software written by Matthew Shepherd, Ryan Mitchell, and
//                  Hrayr Matevosyan at Indiana University, Bloomington
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice and author attribution, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice and author attribution, this list of conditions and the
//    following disclaimer in the documentation and/or other materials
//    provided with the distribution.
// 3. Neither the name of the University nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// Creation of derivative forms of this software for commercial
// utilization may be subject to restriction; written permission may be
// obtained from the Trustees of Indiana University.
//
// INDIANA UNIVERSITY AND THE AUTHORS MAKE NO REPRESENTATIONS OR WARRANTIES,
// EXPRESS OR IMPLIED.  By way of example, but not limitation, INDIANA
// UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES OF MERCANTABILITY OR
// FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF THIS SOFTWARE OR
// DOCUMENTATION WILL NOT INFRINGE ANY PATENTS, COPYRIGHTS, TRADEMARKS,
// OR OTHER RIGHTS.  Neither Indiana University nor the authors shall be
// held liable for any liability with respect to any claim by the user or
// any other party arising from use of the program.
//******************************************************************************

#include <iostream>
#include <utility>
#include <string>
#include <vector>
#include <map>
#include <complex>
#include <fstream>

#include "MinuitInterface/MinuitMinimizationManager.h"
#include "IUAmpTools/IntensityManager.h"
#include "IUAmpTools/Amplitude.h"
#include "IUAmpTools/AmpVecs.h"
#include "IUAmpTools/Neg2LnLikContrib.h"
#include "IUAmpTools/Neg2LnLikContribManager.h"
#include "IUAmpTools/Kinematics.h"
#include "IUAmpTools/NormIntInterface.h"
#include "IUAmpTools/ConfigFileParser.h"
#include "IUAmpTools/ConfigurationInfo.h"
#include "IUAmpTools/ParameterManager.h"
#include "IUAmpTools/LikelihoodCalculator.h"
#include "IUAmpTools/GradientCalculator.h"

class FitResults;

using namespace std;

/**
 * A class to set up and interface with other classes within IUAmpTools.
 *
 * The constructor takes a pointer to a ConfigurationInfo object;
 * from there, all the major classes of IUAmpTools are configured and
 * can be directly accessed via (non-const) pointers through
 * member functions.
 *
 * To reconfigure the AmpToolsInterface class with a new ConfigurationInfo
 * object, use the resetConfigurationInfo method.  This will recreate
 * all the major classes of IUAmpTools.
 *
 * Methods such as loadEvents, processEvents, decayAmplitude, and
 * intensity are provided to perform calculations manually, which can be
 * useful when generating Monte Carlo, for example.
 *
 * For MPI implementations, use the AmpToolsInterfaceMPI class, which
 * is derived from this class.
 *
 * \ingroup IUAmpTools
 */


class AmpToolsInterface{

public:

  enum FunctionalityFlag { kFull, kMCGeneration, kPlotGeneration };

  AmpToolsInterface( FunctionalityFlag flag = kFull );

  /** Constructor.
   * Constructs an AmpToolsInterface
   *
   * \param[in] cfgInfo a pointer to a ConfigurationInfo object;  IUAmpTools
   * classes are set up and configured based on the information contained in this
   * ConfigurationInfo object.
   */

  AmpToolsInterface( ConfigurationInfo* cfgInfo, FunctionalityFlag flag = kFull );

  /** Destructor.
   */

  virtual ~AmpToolsInterface() { clear(); }

  /** Static function to register a user Amplitude class.  For example,
   *  to register a user-defined BreitWigner amplitude, one would use:
   *     AmpToolsInterface::registerAmplitude(BreitWigner());
   */

  static void registerAmplitude( const Amplitude& defaultAmplitude);

  /** Static function to register a user Neg2LnLikContrib class.  For example,
   *  to register a user-defined phaseshift Neg2LnLikContrib, one would use:
   *     AmpToolsInterface::registerAmplitude(phaseshift());
   */

  static void registerNeg2LnLikContrib( const Neg2LnLikContrib& defaulNeg2LnLikContrib);

  /** Static function to register a user DataReader class.  For example,
   *  to register a user-defined CLEODataReader, one would use:
   *     AmpToolsInterface::registerDataReader(CLEODataReader());
   */

  static void registerDataReader( const DataReader& defaultDataReader);

  static void setRandomSeed( unsigned int seed ) { m_randomSeed = seed; }

  /** Use this method to re-initialize all IUAmpTools classes based on information
   *  in a new or modified ConfigurationInfo object.
   */

  void resetConfigurationInfo(ConfigurationInfo* cfgInfo);


  /** Return the ConfigurationInfo object stored in this class.
   *  After modifying this object, one can use the resetConfigurationInfo method
   *  to re-initialize the IUAmpTools classes.
   */

  ConfigurationInfo* configurationInfo() const
  { return m_configurationInfo; }


  /** Pointer to the MinuitMinimizationManager.
   *  Use the methods in MinuitMinimizationManager to do fits.
   */

  MinuitMinimizationManager* minuitMinimizationManager() const
  { return m_minuitMinimizationManager; }

  /** Pointer to the ParameterManager.
   *  Use this to get fit results, for example.
   */

  ParameterManager*          parameterManager() const
  { return m_parameterManager;}


  /** Pointer to the GradientCalculator.
   *  Use this to get the gradient of the likelihood function.
   */
  GradientCalculator* gradientCalculator() const { return m_gradientCalculator; }

  /** Return a map to all Normalization Integral interfaces.
  */
  map< string, NormIntInterface* > normIntMap() const { return m_normIntMap; }

  /** Pointer to an IntensityManager.  There is one for each defined reaction.
   *  (Most applications will not likely need to access these.)
   */

  IntensityManager*     intensityManager     (const string& reactionName) const;


  /** Returns a pointer to a DataReader (for data).
   *  There is one for each reaction.
   */

  DataReader*           dataReader           (const string& reactionName) const;

  /** Returns a pointer to a DataReader (for background).
   *  There is one for each reaction.
   */

  DataReader*           bkgndReader           (const string& reactionName) const;

  /** Returns a pointer to a DataReader (for accepted Monte Carlo).
   *  There is one for each reaction.
   */

  DataReader*           accMCReader          (const string& reactionName) const;


  /** Returns a pointer to a DataReader (for generated Monte Carlo).
   *  There is one for each reaction.
   */

  DataReader*           genMCReader          (const string& reactionName) const;


  /** Pointer to a NormIntInterface (one per reaction).
   *  Use this to access normalization integrals.
   */

  NormIntInterface*     normIntInterface     (const string& reactionName) const;

  /** Pointer to a FitResults class.
   * Use this to access information that is stored after finalizeFit() is called.
   */

  const FitResults* fitResults() const { return m_fitResults; }


  /** Pointer to a Likelihood calculator (one per reaction).
   *  (Most applications will not likely need to access these.)
   *  One can also access the likelihood value through the likelihood methods
   *  below.
   *
   *  \see likelihood
   */

  LikelihoodCalculator* likelihoodCalculator (const string& reactionName) const;


  /** Return the total -2ln(likelihood) (summed over all reactions) using
   *  the parameters stored in the ParameterManager and the amplitudes
   *  in the AmplitudeManagers.  Call this before performing a fit to get
   *  the pre-fit likelihood value and call it after the fit to get the
   *  post-fit value.
   */

  double likelihood() const;

  /** Return the -2ln(likelihood) associated with a particular reaction.
   */

  double likelihood(const string& reactionName) const;

  /** Returns the current value of the likelihood() function and the
   * gradient of the likelihood function with respect to the parameters.
  */
  pair< double, vector<double> > likelihoodAndGradient();

  /** Reinitialize production and amplitude parameters to the values from
   * ConfigurationInfo.  This is useful for repeated fits where most parameters
   * should be seeded with the same values, for example in a likelihood
   * scan over a single parameter.
   */

  void reinitializePars( );

  /** This function will randomly set the production parameters in
   * the likelihood calculator.  It is useful when searching for multiple
   * ambiguous solutions.  Production parameters will be set such that
   * starting fit fraction varies between zero and fraction specified
   * by the optional argument.  The phase will be set randomly between
   * zero and 2 pi.
   */

  void randomizeProductionPars( float maxFitFraction = 1 );

  /** This function will fill the value of a parameter named parName with
   * a randon number between min (0 default) and max (1 default).  The parName
   * should be the name of a parameter declared in the config file using
   * the parameter keyword.  This function is not able to modify parameters
   * that are production coefficients of amplitudes.
   */

  void randomizeParameter( const string& parName, float min = 0, float max = 1 );

  /** Print final fit results to a file.  The tag can be used to
   *  generate a unique name in the case that multiple results are
   *  written for a singele fit job.
   */

  virtual void finalizeFit( const string& tag = "", bool bSaveAmps = false );


  /** For manual calculations:  clear all events and calculations.
   *  Call this before loading events from a new reaction or to start
   *  new calcuations.
   *
   * \param[in] iDataSet used to index simultaneous manual calculations
   *
   *  \see loadEvent
   *  \see loadEvents
   *  \see processEvents
   */

  void clearEvents(unsigned int iDataSet = 0);

  /** For manual calculations:  load event kinematics from a DataReader.
   *  Call this after clearing old events (clearEvents) and before
   *  performing calculations (processEvents).
   *
   * \param[in] iDataSet used to index simultaneous manual calculations
   *
   *  \see clearEvents
   *  \see processEvents
   */

  void loadEvents(DataReader* dataReader,
                  unsigned int iDataSet = 0);

  /** For manual calculations:  load kinematics one event at a time.
   *  Clear old events (clearEvents) before loading a new set of events.
   *  Load all events before performing calculations (processEvents).
   *
   * \param[in] kin the kinematics to be loaded
   * \param[in] iEvent the number of this event (between 0 and nEventsTotal-1)
   * \param[in] nEventsTotal the total number of events to be loaded
   * \param[in] iDataSet used to index simultaneous manual calculations
   *
   *  \see clearEvents
   *  \see processEvents
   */

  void loadEvent(Kinematics* kin, int iEvent = 0, int nEventsTotal = 1,
                 unsigned int iDataSet = 0);

  /** For manual calculations:  perform all calculations on the loaded events.
   *  Load all events (loadEvent or loadEvents) before performing calculations.
   *  Returns the maximum intensity.
   *
   * \param[in] reactionName the name of the reaction that will be calculated
   * \param[in] iDataSet used to index simultaneous manual calculations
   *
   *  \see clearEvents
   *  \see loadEvent
   *  \see loadEvents
   */

  double processEvents(string reactionName,
                       unsigned int iDataSet = 0);


  /** The number of events that have been loaded for manual calculations.
   *
   *  \see clearEvents
   *  \see loadEvent
   *  \see loadEvents
   *  \see processEvents
   */

  int numEvents(unsigned int iDataSet = 0) const;
  double sumWeights(unsigned int iDataSet = 0) const;


  /** Retrieve the intensity calculated for the specified event.
   *  This requires a prior call to processEvents.
   *
   *  \see clearEvents
   *  \see loadEvent
   *  \see loadEvents
   *  \see processEvents
   */

  double intensity(int iEvent,
                   unsigned int iDataSet = 0) const;


  /** Perform an alternate calculation of the intensity.  If there are no
   *  precision problems, this should be identical to the value returned
   *  by the intensity method.
   *  This requires a prior call to processEvents.
   *
   *  \see clearEvents
   *  \see loadEvent
   *  \see loadEvents
   *  \see processEvents
   */

  double alternateIntensity(int iEvent,
                            unsigned int iDataSet = 0) const;


  /** Retrieve a decay amplitude calculated for the specified event.
   *  This requires a prior call to processEvents.
   *
   *  \param[in] ampName  this is the full amplitude name
   *  (as it appears in AmplitudeInfo::fullName)
   *
   *  \see clearEvents
   *  \see loadEvent
   *  \see loadEvents
   *  \see processEvents
   */


  complex<double> decayAmplitude (int iEvent, string ampName,
                                  unsigned int iDataSet = 0) const;

  // complex<double> decayAmplitudeAndWeight (int iEvent, string ampName,
  //                                         unsigned int iDataSet = 0) const;

  void saveAmps( ReactionInfo* reaction, std::vector<string>& saved_files, string suffix );
  void saveAmpVecsToTree( AmpVecs& m_ampvec, string fname, string suffix );


  /** Retrieve the production amplitude associated with a given amplitude.
   *  Recall that all amplitudes are defined as
   *    (production amplitude)  x  (decay amplitude)
   *
   *  The production amplitude that is returned is scaled by the scale
   *  factor that is set for that amplitude (if one has been set).
   *
   *  \param[in] ampName  this is the full amplitude name
   *  (as it appears in AmplitudeInfo::fullName)
   *
   * \see AmplitudeManager::getScale
   */

  complex<double> scaledProductionAmplitude (string ampName,
                                             unsigned int iDataSet = 0) const;


  /** Return the kinematics for a specified event as it was loaded using
   *  the loadEvent or loadEvents method.
   *
   *  \see clearEvents
   *  \see loadEvent
   *  \see loadEvents
   *  \see processEvents
   */

  Kinematics* kinematics(int iEvent,
                         unsigned int iDataSet = 0);


  /** For debugging and diagnostics:  print event kinematics to the screen in
   *  a readable format.
   */

  void printKinematics   (string reactionName, Kinematics* kin) const;


  /** For debugging and diagnostics:  print amplitude values to the screen in
   *  a readable format.
   */

  void printAmplitudes   (string reactionName, Kinematics* kin) const;

  /** For debugging and diagnostics:  print an intensity to the screen in
   *  a readable format.
   */

  void printIntensity    (string reactionName, Kinematics* kin) const;

  /** For debugging and diagnostics:  print event details to the screen in
   *  a readable format.
   */

  void printEventDetails (string reactionName, Kinematics* kin) const;

  /**
   * This allows the user to control how this flag is set on all the IntensityManagers in the fit
   */
  void forceUserVarRecalculation( bool state );

  void print_boundPtrCache();

protected:

  AmpToolsInterface( const AmpToolsInterface& ati );
  AmpToolsInterface& operator=( AmpToolsInterface& ati );

  FunctionalityFlag m_functionality;

  void clear();
  // has to be called by destructor, print_boundPtrCache will crash if called by clear in resetConfigurationInfo
  //   probably because MinuitMinimizationManager isn't created yet
  void clear_and_print();

  ConfigurationInfo*          m_configurationInfo;
  MinuitMinimizationManager*  m_minuitMinimizationManager;
  ParameterManager*           m_parameterManager;
  GradientCalculator*         m_gradientCalculator;

  vector<IntensityManager*>  m_intensityManagers;


  map<string,DataReader*> m_dataReaderMap;
  map<string,DataReader*> m_bkgndReaderMap;
  map<string,DataReader*> m_genMCReaderMap;
  map<string,DataReader*> m_accMCReaderMap;

  // use a set here because entries can't be duplicated and
  // this makes it easy to avoid a double-delete at cleanup time
  set<DataReader*> m_uniqueDataSets;

  map<string,NormIntInterface*>     m_normIntMap;
  map<string,LikelihoodCalculator*> m_likCalcMap;

  static vector<Amplitude*>  m_userAmplitudes;
  static vector<Neg2LnLikContrib*>  m_userNeg2LnLikContribs;
  static vector<DataReader*> m_userDataReaders;

  static unsigned int m_randomSeed;

  // these variables are used in cases where the AmpToolsInterface
  // must provide a place to load the data -- during normal
  // fitting the data are held by the likelihood calculator

  static const int MAXAMPVECS = 50;
  AmpVecs m_ampVecs[MAXAMPVECS];
  string  m_ampVecsReactionName[MAXAMPVECS];
  // static const int MAXAMPVECS = 50;
  // vector<AmpVecs> m_ampVecs = vector<AmpVecs>(MAXAMPVECS);
  // vector<string>  m_ampVecsReactionName = vector<string>(MAXAMPVECS);

  FitResults* m_fitResults;

  float random( float randMax ) const;

private:

  void invalidateAmps();

  static const char* kModule;
};



#endif
