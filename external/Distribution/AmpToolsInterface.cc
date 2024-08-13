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


#include "IUAmpTools/AmpToolsInterface.h"
#include "MinuitInterface/MinuitMinimizationManager.h"
#include "IUAmpTools/AmplitudeManager.h"
#include "IUAmpTools/AmpVecs.h"
#include "IUAmpTools/Kinematics.h"
#include "IUAmpTools/NormIntInterface.h"
#include "IUAmpTools/ConfigFileParser.h"
#include "IUAmpTools/ConfigurationInfo.h"
#include "IUAmpTools/ParameterManager.h"
#include "IUAmpTools/LikelihoodCalculator.h"
#include "IUAmpTools/GradientCalculator.h"
#include "IUAmpTools/AmpToolsInterface.h"
#include "IUAmpTools/FitResults.h"

#include "TFile.h"
#include "TTree.h"

#include "IUAmpTools/report.h"

const char* AmpToolsInterface::kModule = "AmpToolsInterface";

vector<Amplitude*> AmpToolsInterface::m_userAmplitudes;
vector<Neg2LnLikContrib*> AmpToolsInterface::m_userNeg2LnLikContribs;
vector<DataReader*> AmpToolsInterface::m_userDataReaders;
unsigned int AmpToolsInterface::m_randomSeed = 0;

AmpToolsInterface::AmpToolsInterface( FunctionalityFlag flag ):
m_functionality( flag ),
m_configurationInfo( NULL ),
m_minuitMinimizationManager(NULL),
m_parameterManager(NULL),
m_gradientCalculator(NULL),
m_fitResults(NULL)
{
  report( DEBUG, kModule ) << "AmpToolsInterface constructor without cfgInfo" << endl;
  srand( m_randomSeed );
  report( DEBUG, kModule ) << "AmpToolsInterface constructor without cfgInfo done" << endl;
}


AmpToolsInterface::AmpToolsInterface(ConfigurationInfo* configurationInfo, FunctionalityFlag flag ):
m_functionality( flag ),
m_configurationInfo(configurationInfo),
m_minuitMinimizationManager(NULL),
m_parameterManager(NULL),
m_gradientCalculator(NULL),
m_fitResults(NULL)
{
  report (DEBUG, kModule ) << "MAXAMPVECS: " << MAXAMPVECS << endl;
  report (DEBUG, kModule ) << "AmpToolsInterface constructor" << endl;
  resetConfigurationInfo(configurationInfo);
  report (DEBUG, kModule ) << "AmpToolsInterface constructor almost done, seeding" << endl;
  srand( m_randomSeed );
  report (DEBUG, kModule ) << "AmpToolsInterface constructor done" << endl;
}


void
AmpToolsInterface::resetConfigurationInfo(ConfigurationInfo* configurationInfo){

  m_configurationInfo = configurationInfo;

  // check sizeof(GDouble)
  if (sizeof(GDouble) != sizeof(double)){
    report( ERROR, kModule ) << "GDouble and double are not the same size!" << endl;
    assert(false);
  }
  else
    report( DEBUG, kModule ) << "GDouble and double are the same size" << endl;

  clear();

  report( DEBUG, kModule ) << "resetting from configuration info" << endl;

  if( m_functionality == kFull ){

    // ************************
    // create a MinuitMinimizationManager
    // ************************
    report( DEBUG, kModule ) << "creating a MinuitMinimizationManager" << endl;
    m_minuitMinimizationManager = new MinuitMinimizationManager(500);
    report( DEBUG, kModule ) << "  + created a MinuitMinimizationManager" << endl;
  }

  // ************************
  // create an AmplitudeManager for each reaction
  // ************************

  for (unsigned int irct = 0; irct < m_configurationInfo->reactionList().size(); irct++){

    ReactionInfo* reaction = m_configurationInfo->reactionList()[irct];
    string reactionName(reaction->reactionName());
    report( DEBUG, kModule ) << "creating an AmplitudeManager for reaction " << reactionName << endl;

    AmplitudeManager* ampMan = new AmplitudeManager(reaction->particleList(),reactionName);
    report( DEBUG, kModule ) << "  + created an AmplitudeManager for reaction " << reactionName << endl;
    for (unsigned int i = 0; i < m_userAmplitudes.size(); i++){
      ampMan->registerAmplitudeFactor( *m_userAmplitudes[i] );
      report( DEBUG, kModule ) << "  + registered an AmplitudeFactor for reaction " << reactionName << endl;
    }
    ampMan->setupFromConfigurationInfo( m_configurationInfo );
    report( DEBUG, kModule ) << "  + setup AmplitudeManager for reaction " << reactionName << endl;

    if( m_functionality == kFull ){
      ampMan->setOptimizeParIteration( true );
      ampMan->setFlushFourVecsIfPossible( true );
      report( DEBUG, kModule ) << "  + set kFull functionality options for reaction " << reactionName << endl;
    }

    if( m_functionality == kMCGeneration ){
      ampMan->setOptimizeParIteration( false );
      ampMan->setFlushFourVecsIfPossible( false );
      ampMan->setForceUserVarRecalculation( true );
      report( DEBUG, kModule ) << "  + set kMCGeneration functionality options for reaction " << reactionName << endl;
    }

    m_intensityManagers.push_back(ampMan);
    report (DEBUG, kModule ) << "  + added AmplitudeManager to intensity manager list" << endl;
  }

  Neg2LnLikContribManager* lhcontMan = new Neg2LnLikContribManager();
  if( m_functionality == kFull ){
    for (unsigned int i = 0; i < m_userNeg2LnLikContribs.size(); i++){
      lhcontMan->registerNeg2LnLikContrib( *m_userNeg2LnLikContribs[i] );
    }
    Neg2LnLikContrib::setMinimizationManager( m_minuitMinimizationManager );
    lhcontMan->setupFromConfigurationInfo( m_configurationInfo );
  }

  if( m_functionality == kFull ){

    // ************************
    // create a ParameterManager
    // ************************

    m_parameterManager = new ParameterManager ( m_minuitMinimizationManager, m_intensityManagers );
    m_parameterManager->setNeg2LnLikContribManager( lhcontMan );
    m_parameterManager->setupFromConfigurationInfo( m_configurationInfo );

    vector< MinuitParameter* > parValueList = m_parameterManager->getParValueList();
    m_gradientCalculator = new GradientCalculator( parValueList );
    m_gradientCalculator->setFCN([this](){ return this->likelihood(); });
  }

  // ************************
  // loop over reactions
  // ************************

  for (unsigned int irct = 0; irct < m_configurationInfo->reactionList().size(); irct++){

    ReactionInfo* reaction = m_configurationInfo->reactionList()[irct];
    string reactionName(reaction->reactionName());
    IntensityManager* intenMan = intensityManager(reactionName);

    if (!intenMan)
      report( WARNING, kModule ) << "not creating an AmplitudeManager for reaction " << reactionName << endl;


    if( m_functionality == kFull || m_functionality == kPlotGeneration ){

      // ************************
      // create DataReaders
      // ************************

      for (unsigned int i = 0; i < m_userDataReaders.size(); i++){
        if (reaction->data().first == m_userDataReaders[i]->name()){
          report( DEBUG, kModule ) << "creating DataReaders for reaction " << reactionName << " with args: ";
          for (unsigned int j = 0; j < reaction->data().second.size(); j++){
            report( DEBUG, kModule ) << reaction->data().second[j] << " ";
          }
          report( DEBUG, kModule ) << endl;
          m_dataReaderMap[reactionName] = m_userDataReaders[i]->newDataReader(reaction->data().second);
        }
        if (reaction->bkgnd().first == m_userDataReaders[i]->name()){
          report( DEBUG, kModule ) << "creating DataReaders for reaction " << reactionName << " with args: ";
          for (unsigned int j = 0; j < reaction->bkgnd().second.size(); j++){
            report( DEBUG, kModule ) << reaction->bkgnd().second[j] << " ";
          }
          report( DEBUG, kModule ) << endl;
          m_bkgndReaderMap[reactionName] = m_userDataReaders[i]->newDataReader(reaction->bkgnd().second);
        }
        if (reaction->genMC().first == m_userDataReaders[i]->name()){
          report( DEBUG, kModule ) << "creating DataReaders for reaction " << reactionName << " with args: ";
          for (unsigned int j = 0; j < reaction->genMC().second.size(); j++){
            report( DEBUG, kModule ) << reaction->genMC().second[j] << " ";
          }
          report( DEBUG, kModule ) << endl;
          m_genMCReaderMap[reactionName] = m_userDataReaders[i]->newDataReader(reaction->genMC().second);
        }
        if (reaction->accMC().first == m_userDataReaders[i]->name()){
          report( DEBUG, kModule ) << "creating DataReaders for reaction " << reactionName << " with args: ";
          for (unsigned int j = 0; j < reaction->accMC().second.size(); j++){
            report( DEBUG, kModule ) << reaction->accMC().second[j] << " ";
          }
          report( DEBUG, kModule ) << endl;
          m_accMCReaderMap[reactionName] = m_userDataReaders[i]->newDataReader(reaction->accMC().second);
        }
      }

      report( DEBUG, kModule ) << "current size of m_dataReaderMap: " << m_dataReaderMap.size() << endl;
      report( DEBUG, kModule ) << "current size of m_bkgndReaderMap: " << m_bkgndReaderMap.size() << endl;
      report( DEBUG, kModule ) << "current size of m_genMCReaderMap: " << m_genMCReaderMap.size() << endl;
      report( DEBUG, kModule ) << "current size of m_accMCReaderMap: " << m_accMCReaderMap.size() << endl;

      DataReader* dataRdr  =  dataReader(reactionName);
      DataReader* bkgndRdr = bkgndReader(reactionName);
      DataReader* genMCRdr = genMCReader(reactionName);
      DataReader* accMCRdr = accMCReader(reactionName);

      report( DEBUG, kModule ) << "creating DataReaders for reaction " << reactionName << endl;
      report( DEBUG, kModule ) << "  + dataRdr:  " << (dataRdr ? dataRdr->identifier()  : "NULL") << endl;
      report( DEBUG, kModule ) << "  + bkgndRdr: " << (bkgndRdr ? bkgndRdr->identifier() : "NULL") << endl;
      report( DEBUG, kModule ) << "  + genMCRdr: " << (genMCRdr ? genMCRdr->identifier() : "NULL") << endl;
      report( DEBUG, kModule ) << "  + accMCRdr: " << (accMCRdr ? accMCRdr->identifier() : "NULL") << endl;

      m_uniqueDataSets.insert( dataRdr );
      m_uniqueDataSets.insert( bkgndRdr );
      m_uniqueDataSets.insert( genMCRdr );
      m_uniqueDataSets.insert( accMCRdr );

      report( DEBUG, kModule ) << "  + current size of m_uniqueDataSets: " << m_uniqueDataSets.size() << endl;

      if (!dataRdr)
        report( WARNING, kModule ) << "not creating a DataReader for data associated with reaction " << reactionName << endl;
      if (!genMCRdr)
        report( WARNING, kModule ) << "not creating a DataReader for generated MC associated with reaction " << reactionName << endl;
      if (!accMCRdr)
        report( WARNING, kModule ) << "not creating a DataReader for accepted MC associated with reaction " << reactionName << endl;

      if( m_functionality == kFull ){

        // ************************
        // create a NormIntInterface
        // ************************

        // note that in the case that the ATI is being used for plot generation
        // then the NI's should be obtained from the FitResults object as this
        // contains the NI cache at the end of the fit

        NormIntInterface* normInt = NULL;
        if (genMCRdr && accMCRdr && intenMan && !(reaction->normIntFileInput())){
          report( DEBUG, kModule ) << "creating new NormIntInterface:" << endl;
          report( DEBUG, kModule ) << "  + genMCRdr: " << genMCRdr->identifier() << endl;
          report( DEBUG, kModule ) << "  + accMCRdr: " << accMCRdr->identifier() << endl;
          normInt = new NormIntInterface(genMCRdr, accMCRdr, *intenMan);
          m_normIntMap[reactionName] = normInt;
          if (reaction->normIntFile() == "")
            report( WARNING, kModule ) << "no name given to NormInt file for reaction " << reactionName << endl;
        }
        else if (reaction->normIntFileInput()){

          normInt = new NormIntInterface(reaction->normIntFile());
          m_normIntMap[reactionName] = normInt;
        }
        else{

          report( WARNING, kModule ) << "not creating a NormIntInterface for reaction " << reactionName << endl;
        }

        // ************************
        // create a LikelihoodCalculator
        // ************************

        LikelihoodCalculator* likCalc = NULL;
        if (intenMan && normInt && dataRdr && m_parameterManager){
          likCalc = new LikelihoodCalculator(*intenMan, *normInt, dataRdr, bkgndRdr, *m_parameterManager);
          m_likCalcMap[reactionName] = likCalc;
        }
        else{
          report( WARNING, kModule ) << "not creating a LikelihoodCalculator for reaction " << reactionName << endl;
        }
      }
    }
  }

  // ************************
  // create FitResults
  // ************************

  if( m_functionality == kFull ){

    m_fitResults = new FitResults( m_configurationInfo,
                                  m_intensityManagers,
                                  m_likCalcMap,
                                  m_normIntMap,
                                  m_minuitMinimizationManager,
                                  m_parameterManager );
  }

  // if functionality is for PlotGeneration then fitResults
  // is left NULL -- in general m_fitResults is used to keep track
  // of an output of the interface
}



double
AmpToolsInterface::likelihood( const string& reactionName ) const {
  LikelihoodCalculator* likCalc = likelihoodCalculator(reactionName);
  if (likCalc) return (*likCalc)();
  return 0.0;
}


double
AmpToolsInterface::likelihood() const {
  if( m_minuitMinimizationManager ){
    m_minuitMinimizationManager->parameterManager().synchronizeMinuit();
    return m_minuitMinimizationManager->evaluateFunction();
  }
  return 0.0;
}

pair< double, vector<double> >
AmpToolsInterface::likelihoodAndGradient(){
  m_gradientCalculator->calculate();
  return pair< double, vector<double> >( m_gradientCalculator->fcn(), m_gradientCalculator->grad_fcn() );
}

void
AmpToolsInterface::reinitializePars(){

  // is this fully robust for MPI?  what if it is called on lead only? invalidate will be problematic?

  // shouldn't be callin' unless you're fittin'
  if( m_functionality != kFull ) return;

  // reinitialize production parameters
  vector< AmplitudeInfo* > amps = m_configurationInfo->amplitudeList();
  for( vector< AmplitudeInfo* >::iterator ampItr = amps.begin();
      ampItr != amps.end();
      ++ampItr ){

    if( (**ampItr).fixed() ) continue;

    string ampName = (**ampItr).fullName();
    complex< double > prodPar = (**ampItr).value();

    m_parameterManager->setProductionParameter( ampName, prodPar );
  }

  // reinitialize amplitude parameters
  vector<ParameterInfo*> parInfoVec = m_configurationInfo->parameterList();
  for ( vector< ParameterInfo* >::iterator parItr = parInfoVec.begin(); parItr != parInfoVec.end(); ++parItr ){

    if( (**parItr).fixed() ) continue;

    string parName = (**parItr).parName();
    double value = (**parItr).value();

    m_parameterManager->setAmpParameter( parName, value );
  }

  // reset parameter steps after reinitializing parameters
  minuitMinimizationManager()->resetErrors();

  // reset flags in AmpVecs which will trigger recalculation of all
  // the terms -- this is necessary for example, in cases where
  // pre-calculated user data depends on parameters that might change
  invalidateAmps();
}

void
AmpToolsInterface::randomizeProductionPars( float maxFitFraction ){

  // shouldn't be callin' unless you're fittin'
  if( m_functionality != kFull ) return;

  vector< AmplitudeInfo* > amps = m_configurationInfo->amplitudeList();

  for( vector< AmplitudeInfo* >::iterator ampItr = amps.begin();
      ampItr != amps.end();
      ++ampItr ){

    if( (**ampItr).fixed() ) continue;

    string ampName = (**ampItr).fullName();
    string reac = (**ampItr).reactionName();

    double numSignalEvents = likelihoodCalculator(reac)->numSignalEvents();
    // for the NI interface to use the cache to avoid a mess with MPI jobs and
    // retrigging of NI calculation -- we don't require a precise number
    // for this anyway!
    double normInt = normIntInterface(reac)->normInt(ampName,ampName,true).real();
    double ampScale = intensityManager(reac)->getScale(ampName);

    // fit fraction = ( scale^2 * prodPar^2 * normInt ) / numSignalEvents

    double fitFraction = random( maxFitFraction );
    double prodParMag = sqrt( ( numSignalEvents * fitFraction ) /
                              ( ampScale * ampScale * normInt ) );
    double prodParPhase = ( (**ampItr).real() ? 0 : random( 2*PI ) );

    complex< double > prodPar( prodParMag*cos( prodParPhase ),
                              prodParMag*sin( prodParPhase ) );

    m_parameterManager->setProductionParameter( ampName, prodPar );
  }

  // reset parameter steps after randomizing parameters
  minuitMinimizationManager()->resetErrors();
}

void
AmpToolsInterface::randomizeParameter( const string& parName, float min, float max ){

  vector<ParameterInfo*> parInfoVec = m_configurationInfo->parameterList();

  std::vector<ParameterInfo*>::iterator parItr = parInfoVec.begin();
  for( ; parItr != parInfoVec.end(); ++parItr ){

    if( (**parItr).parName() == parName ) break;
  }

  if( parItr == parInfoVec.end() ){

    report( ERROR, kModule ) << "request to randomize nonexistent parameter:  " << parName << endl;
    return;
  }

  if( (**parItr).fixed() ){

    report( ERROR, kModule ) << "request to randomize a parameter named " << parName
    << " that is fixed.  Ignoring this request." << endl;
    return;
  }

  double value = min + random( max - min );
  m_parameterManager->setAmpParameter( parName, value );

  // reset parameter steps after randomizing parameters
  minuitMinimizationManager()->resetErrors();

  // reset flags in AmpVecs which will trigger recalculation of all
  // the terms -- this is necessary for example, in cases where
  // pre-calculated user data depends on parameters that might change
  invalidateAmps();
}

// Save the complex numbers of the decay amplitudes
void 
AmpToolsInterface::saveAmps(
  ReactionInfo* reaction,
  std::vector<string>& saved_files,
  string suffix=""){

    string reactionName(reaction->reactionName());
    // AmpVecs store complex decay amplitudes. LikelihoodCalculator owns the signal and background
    // NormIntInterface owns the generated and accepted MC
    LikelihoodCalculator* likCalc = likelihoodCalculator(reactionName);
    NormIntInterface* normInt = normIntInterface(reactionName);
    AmpVecs& ampVecsSignal = likCalc->ampVecsSignal();
    AmpVecs& ampVecsGenMC  = normInt->ampVecsGenMC();
    AmpVecs& ampVecsAccMC  = normInt->ampVecsAccMC();
    AmpVecs& ampVecsBkgnd  = likCalc->ampVecsBkgnd();
    std::vector<AmpVecs*> ampVecs = {&ampVecsSignal, &ampVecsGenMC, &ampVecsAccMC, &ampVecsBkgnd};

    // Note. Order must be the same as above
    vector<string> fnames = {};
    fnames.push_back(reaction->data().second[0]);
    fnames.push_back(reaction->genMC().second[0]);
    fnames.push_back(reaction->accMC().second[0]);
    bool bkgndFileExists = reaction->bkgnd().first.empty() == false;
    if (bkgndFileExists) // Only bkgnd file is optional
      fnames.push_back(reaction->bkgnd().second[0]);

    // DUMP ALL COMPLEX AMPLITUDES TO A ROOT FILE
    // NOTE: for the Zlm amplitudes, I think the terms are purely real since we do this 
    //       separation in 4 different coherent sums
    //       Assume it should be complex and reduce the file later in post
    for (int i=0; i<fnames.size(); i++){
      AmpVecs* ampVec = ampVecs[i];
      string fname = fnames[i];
      bool alreadySaved = std::find(saved_files.begin(), saved_files.end(), fname) != saved_files.end();
      if (i==3 && !bkgndFileExists || alreadySaved)
        continue;
      saved_files.push_back(fname);
      if ( fname.substr(fname.size()-5, fname.size()) != ".root" )
        assert( fname.substr(fname.size()-5, fname.size()) == ".root" );
      fname = fname.substr(0, fname.size()-5);
      saveAmpVecsToTree(*ampVec, fname, suffix);
    }
}

// Write the ampvecs object to a root file
void 
AmpToolsInterface::saveAmpVecsToTree(
  AmpVecs& m_ampvec, 
  string fname, 
  string suffix){

    report( DEBUG, kModule ) << "Saving copy of " << fname << " to " << fname+suffix+".root" << endl;

    // TFile *inputFile = new TFile((fname+".root").c_str(), "READ");
    // if (!inputFile || inputFile->IsZombie()) {
    //     std::cerr << "Error opening input file\n";
    //     return;
    // }
    // TTree *inputTree = (TTree*)inputFile->Get("kin");

    // Save as a hidden file. IDK if this is a good idea.
    //   We would very like to post-process the root files and store them in a better
    //   data structure that has better read speed
    TFile *outputFile = new TFile((fname+suffix+".root").c_str(), "RECREATE");
    TTree *outputTree = new TTree("kin", "kin");

    int iNTerms = m_ampvec.m_iNTerms;
    vector<string>& ampNames = m_ampvec.m_termNames;

    // Replace all occurrences of "::" with "_"
    for (auto& ampName : ampNames) {
        size_t pos;
        while ((pos = ampName.find("::")) != std::string::npos) {
            ampName.replace(pos, 2, ".");
        }
    }

    double weight;
    std::vector<double> ta_re(iNTerms);
    std::vector<double> ta_im(iNTerms);
    for (int iAmp=0; iAmp<iNTerms; iAmp++){
      string ampName = ampNames[iAmp];
      outputTree->Branch((ampName+"_re").c_str(), &ta_re[iAmp], (ampName+"_re/D").c_str());
      outputTree->Branch((ampName+"_im").c_str(), &ta_im[iAmp], (ampName+"_im/D").c_str());
    }
    outputTree->Branch("weight", &weight, "weight/D");

    int m_iNEvents = m_ampvec.m_iNEvents;
    for (int iEvent=0; iEvent<m_iNEvents; iEvent++){

      weight = m_ampvec.m_pdWeights[iEvent];
      for (int iAmp=0; iAmp<iNTerms; iAmp++){
        ta_re[iAmp] = m_ampvec.m_pdAmps[2*m_iNEvents*iAmp+2*iEvent];
        ta_im[iAmp] = m_ampvec.m_pdAmps[2*m_iNEvents*iAmp+2*iEvent+1];
      }
      outputTree->Fill();
    }

    outputFile->Write();
    outputFile->Close();
}


void
AmpToolsInterface::finalizeFit( const string& tag ){

  // ************************
  // save fit parameters
  // ************************

  if (tag != ""){
    m_fitResults->saveResults();
    m_fitResults->writeResults( m_configurationInfo->fitOutputFileName( tag ) );
  }

  // ************************
  // save normalization integrals
  // ************************

  for (unsigned int irct = 0; irct < m_configurationInfo->reactionList().size(); irct++){

    ReactionInfo* reaction = m_configurationInfo->reactionList()[irct];

    if( !reaction->normIntFileInput() ){
      // cout << "Exporting normInt cache for reaction " << reaction->reactionName() << endl;
      string reactionName(reaction->reactionName());
      NormIntInterface* normInt = normIntInterface(reactionName);

      if (tag == ""){
        // the call to FitResults::writeResults will force a cache update
        // there is no need to do it twice
        if (normInt->hasAccessToMC())
          normInt->forceCacheUpdate();
      }
      normInt->exportNormIntCache( reaction->normIntFile() );
    }
  }

  // Do this after dumping normalization integrals since we need to force cache update first
  string suffix("_amps");
  std::vector<string> saved_files = {};
  for (unsigned int irct = 0; irct < m_configurationInfo->reactionList().size(); irct++){
    ReactionInfo* reaction = m_configurationInfo->reactionList()[irct];
    saveAmps( reaction, saved_files, suffix );
  }
}

IntensityManager*
AmpToolsInterface::intensityManager(const string& reactionName) const {
  for (unsigned int i = 0; i < m_intensityManagers.size(); i++){
    if (m_intensityManagers[i]->reactionName() == reactionName)
      return m_intensityManagers[i];
  }
  return (IntensityManager*) NULL;
}


DataReader*
AmpToolsInterface::dataReader (const string& reactionName) const {
  if (m_dataReaderMap.find(reactionName) != m_dataReaderMap.end())
    return m_dataReaderMap.find(reactionName)->second;
  return (DataReader*) NULL;
}

DataReader*
AmpToolsInterface::bkgndReader (const string& reactionName) const {
  if (m_bkgndReaderMap.find(reactionName) != m_bkgndReaderMap.end())
    return m_bkgndReaderMap.find(reactionName)->second;
  return (DataReader*) NULL;
}

DataReader*
AmpToolsInterface::genMCReader (const string& reactionName) const {
  if (m_genMCReaderMap.find(reactionName) != m_genMCReaderMap.end())
    return m_genMCReaderMap.find(reactionName)->second;
  return (DataReader*) NULL;
}


DataReader*
AmpToolsInterface::accMCReader (const string& reactionName) const {
  if (m_accMCReaderMap.find(reactionName) != m_accMCReaderMap.end())
    return m_accMCReaderMap.find(reactionName)->second;
  return (DataReader*) NULL;
}


NormIntInterface*
AmpToolsInterface::normIntInterface (const string& reactionName) const {
  if (m_normIntMap.find(reactionName) != m_normIntMap.end())
    return m_normIntMap.find(reactionName)->second;
  return (NormIntInterface*) NULL;
}


LikelihoodCalculator*
AmpToolsInterface::likelihoodCalculator (const string& reactionName) const {
  if (m_likCalcMap.find(reactionName) != m_likCalcMap.end())
    return m_likCalcMap.find(reactionName)->second;
  return (LikelihoodCalculator*) NULL;
}



void
AmpToolsInterface::registerAmplitude( const Amplitude& amplitude){

  m_userAmplitudes.push_back(amplitude.clone());

}

void
AmpToolsInterface::registerNeg2LnLikContrib( const Neg2LnLikContrib& lhcont){

  m_userNeg2LnLikContribs.push_back(lhcont.clone());

}


void
AmpToolsInterface::registerDataReader( const DataReader& dataReader){

  m_userDataReaders.push_back(dataReader.clone());

}

void
AmpToolsInterface::clear(){
  report( DEBUG, kModule ) << "AmpToolsInterface::clear() called by destructor or resetConfigurationInfo" << endl;
  if( m_configurationInfo != NULL ){

    for (unsigned int irct = 0; irct < m_configurationInfo->reactionList().size(); irct++){

      ReactionInfo* reaction = m_configurationInfo->reactionList()[irct];
      string reactionName(reaction->reactionName());

    if (likelihoodCalculator(reactionName)){
        report( DEBUG, kModule ) << "Deleting m_likCalcMap for reaction " << reactionName << endl;
        delete m_likCalcMap[reactionName];
        m_likCalcMap[reactionName] = nullptr;
      }
      if (normIntInterface(reactionName)){
        report( DEBUG, kModule ) << "Deleting normIntInterface for reaction " << reactionName << endl;
        delete normIntInterface(reactionName);
        m_normIntMap[reactionName] = nullptr;
      }
    }

    // logic above won't work for these since the lookup function for each reaction
    // depends on the reaction itself
    for (unsigned int i = 0; i < m_intensityManagers.size(); i++){
      if (m_intensityManagers[i]){
        report( DEBUG, kModule ) << "Deleting intensityManager for reaction " << m_intensityManagers[i]->reactionName() << endl;
        delete m_intensityManagers[i];
        m_intensityManagers[i] = nullptr;
      }
    }

    report( DEBUG, kModule ) << "Clearing m_uniqueDataSets with size " << m_uniqueDataSets.size() << endl;
    for( std::set<DataReader*>::iterator dataReader = m_uniqueDataSets.begin();
        dataReader != m_uniqueDataSets.end(); ++dataReader ){
        DataReader* reader = *dataReader;
        if (reader){
          report (DEBUG, kModule ) << "Deleting dataReader with ID: " << (reader)->identifier() << endl;
          delete reader;
          report (DEBUG, kModule ) << "  ^- Deleted..." << endl;
        }
    }
  }

  report (DEBUG, kModule ) << "Clearing m_intensityManagers" << endl;
  m_intensityManagers.clear();
  report (DEBUG, kModule ) << "Clearing m_dataReaderMap" << endl;
  m_dataReaderMap.clear();
  report (DEBUG, kModule ) << "Clearing m_genMCReaderMap" << endl;
  m_genMCReaderMap.clear();
  report (DEBUG, kModule ) << "Clearing m_accMCReaderMap" << endl;
  m_accMCReaderMap.clear();
  report (DEBUG, kModule ) << "Clearing m_bkgndReaderMap" << endl;
  m_bkgndReaderMap.clear();
  report (DEBUG, kModule ) << "Clearing m_normIntMap" << endl;
  m_uniqueDataSets.clear();
  report (DEBUG, kModule ) << "Clearing m_normIntMap" << endl;
  m_normIntMap.clear();
  report (DEBUG, kModule ) << "Clearing m_likCalcMap" << endl;
  m_likCalcMap.clear();

  report( DEBUG, kModule ) << "Deallocating ampvecs by clear():" << endl;
  for (unsigned int i = 0; i < MAXAMPVECS; i++){
    m_ampVecs[i].deallocAmpVecs();
    m_ampVecsReactionName[i] = "";
  }

  report (DEBUG, kModule ) << "Deleting gradientCalculator" << endl;
  if (gradientCalculator()){
    delete gradientCalculator();
    m_gradientCalculator = nullptr;
  }

  report (DEBUG, kModule ) << "Deleting parameterManager" << endl;
  if (parameterManager()){
    delete parameterManager();
    m_parameterManager = nullptr;
  }

  report (DEBUG, kModule ) << "Deleting fitResults" << endl;
  if (fitResults()){
    delete fitResults();
    m_fitResults = nullptr;
  }

  report (DEBUG, kModule ) << "Deleting minuitMinimizationManager" << endl;
  if (minuitMinimizationManager()){
    delete minuitMinimizationManager();
    m_minuitMinimizationManager = nullptr;
  }

  // loop over m_userDataReaders and clear tracked instances
  for (unsigned int i = 0; i < m_userDataReaders.size(); i++){
    report (DEBUG, kModule ) << "Clearing userDataReader" << endl;
    m_userDataReaders[i]->clearDataReaderInstances();
  }
}

void
AmpToolsInterface::clear_and_print(){
  report( DEBUG, kModule ) << "AmpToolsInterface::clear() called by destructor or resetConfigurationInfo" << endl;
  if( m_configurationInfo != NULL ){

    for (unsigned int irct = 0; irct < m_configurationInfo->reactionList().size(); irct++){

      ReactionInfo* reaction = m_configurationInfo->reactionList()[irct];
      string reactionName(reaction->reactionName());

      if (likelihoodCalculator(reactionName)){
        print_boundPtrCache();
        report( DEBUG, kModule ) << "Deleting m_likCalcMap for reaction " << reactionName << endl;
        delete m_likCalcMap[reactionName];
      }
      if (normIntInterface(reactionName)){
        print_boundPtrCache();
        report( DEBUG, kModule ) << "Deleting normIntInterface for reaction " << reactionName << endl;
        delete normIntInterface(reactionName);
      }
    }

    // logic above won't work for these since the lookup function for each reaction
    // depends on the reaction itself
    for (unsigned int i = 0; i < m_intensityManagers.size(); i++){
      print_boundPtrCache();
      report( DEBUG, kModule ) << "Deleting intensityManager for reaction " << m_intensityManagers[i]->reactionName() << endl;
      delete m_intensityManagers[i];
    }

    for( std::set<DataReader*>::iterator dataReader = m_uniqueDataSets.begin();
        dataReader != m_uniqueDataSets.end(); ++dataReader ){
      if( *dataReader ){
        print_boundPtrCache();
        report (DEBUG, kModule ) << "Deleting dataReader " << (*dataReader)->name() << endl;
        delete *dataReader;
      }
    }
  }

  print_boundPtrCache();
  report (DEBUG, kModule ) << "Clearing m_intensityManagers" << endl;
  m_intensityManagers.clear();
  print_boundPtrCache();
  report (DEBUG, kModule ) << "Clearing m_dataReaderMap" << endl;
  m_dataReaderMap.clear();
  print_boundPtrCache();
  report (DEBUG, kModule ) << "Clearing m_genMCReaderMap" << endl;
  m_genMCReaderMap.clear();
  print_boundPtrCache();
  report (DEBUG, kModule ) << "Clearing m_accMCReaderMap" << endl;
  m_accMCReaderMap.clear();
  print_boundPtrCache();
  report (DEBUG, kModule ) << "Clearing m_bkgndReaderMap" << endl;
  m_bkgndReaderMap.clear();
  print_boundPtrCache();
  report (DEBUG, kModule ) << "Clearing m_normIntMap" << endl;
  m_uniqueDataSets.clear();
  print_boundPtrCache();
  report (DEBUG, kModule ) << "Clearing m_normIntMap" << endl;
  m_normIntMap.clear();
  print_boundPtrCache();
  report (DEBUG, kModule ) << "Clearing m_likCalcMap" << endl;
  m_likCalcMap.clear();

  print_boundPtrCache();
  report( DEBUG, kModule ) << "Deallocating ampvecs by clear():" << endl;
  for (unsigned int i = 0; i < MAXAMPVECS; i++){
    m_ampVecs[i].deallocAmpVecs();
    m_ampVecsReactionName[i] = "";
  }

  // NOTE:  order matters here -- the ParameterManagers need to be deleted
  // before the MinuitMinimizationManager as some types of parameter constraints
  // like GaussianBound, need to detach themselves from the minimizaiton manager

  if (parameterManager()){
    print_boundPtrCache();
    report (DEBUG, kModule ) << "Deleting parameterManager" << endl;
    delete parameterManager();
  }

  if (fitResults()) {
    print_boundPtrCache();
    report (DEBUG, kModule ) << "Deleting fitResults" << endl;
    delete fitResults();
  }

  if (minuitMinimizationManager()){
    print_boundPtrCache();
    report (DEBUG, kModule ) << "Deleting minuitMinimizationManager" << endl;
    delete minuitMinimizationManager();
  }
}


void
AmpToolsInterface::clearEvents(unsigned int iDataSet){

  if (iDataSet >= MAXAMPVECS){
    report( ERROR, kModule ) << "data set index out of range" << endl;
    assert(false);
  }

  report( DEBUG, kModule ) << "Deallocating ampvecs by clearEvents():" << endl;
  m_ampVecsReactionName[iDataSet] = "";
  m_ampVecs[iDataSet].deallocAmpVecs();

}


void
AmpToolsInterface::loadEvents(DataReader* dataReader,
                              unsigned int iDataSet){

  if (iDataSet >= MAXAMPVECS){
    report( ERROR, kModule ) << "data set index out of range" << endl;
    assert(false);
  }

  clearEvents(iDataSet);

  // if the data reader is null then this will just leave the AmpVecs
  // object in a state where it contains no data, which is consistent
  // with no data reader available (some DataReaders, like those for
  // background) are optional
  if( dataReader != NULL ){
    report( DEBUG, kModule ) << "loading events from data reader " << dataReader->name() << "Dataset Index: " << iDataSet << endl;
    m_ampVecs[iDataSet].loadData(dataReader);
  }
}


void
AmpToolsInterface::loadEvent(Kinematics* kin, int iEvent, int nEventsTotal,
                             unsigned int iDataSet){

  if (iDataSet >= MAXAMPVECS){
    report( ERROR, kModule ) << "data set index out of range" << endl;
    assert(false);
  }

  report( DEBUG, kModule ) << "loading events from Kinematics object" << endl;
  m_ampVecs[iDataSet].loadEvent(kin, iEvent, nEventsTotal);

}


double
AmpToolsInterface::processEvents(string reactionName,
                                 unsigned int iDataSet) {

  if (iDataSet >= MAXAMPVECS){
    report( ERROR, kModule ) << "data set index out of range" << endl;
    assert(false);
  }

  bool isFirstPass = (m_ampVecs[iDataSet].m_pdAmps == 0);

  m_ampVecsReactionName[iDataSet] = reactionName;

  IntensityManager* intenMan = intensityManager(reactionName);

  if (isFirstPass) m_ampVecs[iDataSet].allocateTerms(*intenMan,true);

  return intenMan->calcIntensities(m_ampVecs[iDataSet]);

}


int
AmpToolsInterface::numEvents(unsigned int iDataSet) const {

  if (iDataSet >= MAXAMPVECS){
    report( ERROR, kModule ) << "data set index out of range" << endl;
    assert(false);
  }

  return m_ampVecs[iDataSet].m_iNTrueEvents;

}

double
AmpToolsInterface::sumWeights(unsigned int iDataSet) const {

  if (iDataSet >= MAXAMPVECS){
    report( ERROR, kModule ) << "data set index out of range" << endl;
    assert(false);
  }

  return m_ampVecs[iDataSet].m_dSumWeights;

}


Kinematics*
AmpToolsInterface::kinematics(int iEvent,
                              unsigned int iDataSet){

  if (iDataSet >= MAXAMPVECS){
    report( ERROR, kModule ) << "data set index out of range" << endl;
    assert(false);
  }

  return m_ampVecs[iDataSet].getEvent(iEvent);

}


double
AmpToolsInterface::intensity(int iEvent,
                             unsigned int iDataSet) const {

  if (iDataSet >= MAXAMPVECS){
    report( ERROR, kModule ) << "data set index out of range" << endl;
    assert(false);
  }

  if (iEvent >= m_ampVecs[iDataSet].m_iNTrueEvents || iEvent < 0){
    report( ERROR, kModule ) << "out of bounds in intensity call" << endl;
    assert(false);
  }

  return m_ampVecs[iDataSet].m_pdIntensity[iEvent];

}


complex<double>
AmpToolsInterface::decayAmplitude (int iEvent, string ampName,
                                   unsigned int iDataSet) const {

  if (iDataSet >= MAXAMPVECS){
    report( ERROR, kModule ) << "data set index out of range" << endl;
    assert(false);
  }

  if (iEvent >= m_ampVecs[iDataSet].m_iNTrueEvents || iEvent < 0){
    report( ERROR, kModule ) << "out of bounds in decafinyAmplitude call" << endl;
    assert(false);
  }

  IntensityManager* intenMan = intensityManager(m_ampVecsReactionName[iDataSet]);

  int iAmp = intenMan->termIndex(ampName);

  // fix!! this experession is not generally correct for all intensity
  // managers -- put as helper function in AmpVecs?

  return complex<double>
  (m_ampVecs[iDataSet].m_pdAmps[2*m_ampVecs[iDataSet].m_iNEvents*iAmp+2*iEvent],
    m_ampVecs[iDataSet].m_pdAmps[2*m_ampVecs[iDataSet].m_iNEvents*iAmp+2*iEvent+1]);

}

// pair< double, complex<double> >
// AmpToolsInterface::decayAmplitudeAndWeight const(
//   int iEvent, string ampName,
//   unsigned int iDataSet) const {

//   if (iDataSet >= MAXAMPVECS){
//     report( ERROR, kModule ) << "data set index out of range" << endl;
//     assert(false);
//   }

//   if (iEvent >= m_ampVecs.m_iNTrueEvents || iEvent < 0){
//     report( ERROR, kModule ) << "out of bounds in decayAmplitude call" << endl;
//     assert(false);
//   }

//   complex<double> amp = decayAmplitude(iEvent, ampName, iDataSet);
//   double weight = m_ampVecs.m_pdWeights[iEvent];

//   return pair< double, complex<double> >( weight, amp );
// }


complex<double>
AmpToolsInterface::scaledProductionAmplitude (string ampName, unsigned int iDataSet) const {

  const IntensityManager* intenMan = intensityManager(m_ampVecsReactionName[iDataSet]);

  double scale = intenMan->getScale( ampName );
  complex< double > prodAmp = intenMan->productionFactor( ampName );

  return scale * prodAmp;
}


double
AmpToolsInterface::alternateIntensity(int iEvent,
                                      unsigned int iDataSet) const {

  if (iDataSet >= MAXAMPVECS){
    report( ERROR, kModule ) << "data set index out of range" << endl;
    assert(false);
  }

  double runningIntensity = 0.0;

  // loop over sums

  vector<CoherentSumInfo*> sums = m_configurationInfo->coherentSumList(m_ampVecsReactionName[iDataSet]);
  for (unsigned int iSum = 0; iSum < sums.size(); iSum++){

    complex<double> runningAmplitude(0.0,0.0);

    // loop over amps

    vector<AmplitudeInfo*> amps =
    m_configurationInfo->amplitudeList(m_ampVecsReactionName[iDataSet],sums[iSum]->sumName());
    for (unsigned int iAmp = 0; iAmp < amps.size(); iAmp++){

      complex<double> P = scaledProductionAmplitude(amps[iAmp]->fullName());
      complex<double> D = decayAmplitude(iEvent,amps[iAmp]->fullName(),iDataSet);

      runningAmplitude += P*D;

    }

    runningIntensity += norm(runningAmplitude);

  }

  return runningIntensity;
}

void
AmpToolsInterface::printKinematics(string reactionName, Kinematics* kin) const {

  ReactionInfo* reaction = m_configurationInfo->reaction(reactionName);
  vector<TLorentzVector> momenta = kin->particleList();

  if (reaction->particleList().size() != momenta.size()){
    report( ERROR, kModule ) << "kinematics incompatible with this reaction" << endl;
    assert(false);
  }

  report( INFO, kModule ) << "  +++++++++++++++++++++++++++++++++" << endl;
  report( INFO, kModule ) << "    EVENT KINEMATICS " << endl;
  streamsize defaultStreamSize = cout.precision(15);
  for (unsigned int imom = 0; imom < momenta.size(); imom++){
    report( INFO, kModule ) << "      particle " << reaction->particleList()[imom] << endl;
    report( INFO, kModule ) << "          E  = " << momenta[imom].E() << endl;
    report( INFO, kModule ) << "          Px = " << momenta[imom].Px() << endl;
    report( INFO, kModule ) << "          Py = " << momenta[imom].Py() << endl;
    report( INFO, kModule ) << "          Pz = " << momenta[imom].Pz() << endl;
  }
  cout.precision(defaultStreamSize);
  report( INFO, kModule ) << "  +++++++++++++++++++++++++++++++++" << endl << endl;
}


void
AmpToolsInterface::printAmplitudes(string reactionName, Kinematics* kin) const {

  IntensityManager* intenMan = intensityManager(reactionName);

  if( intenMan->type() != IntensityManager::kAmplitude ){

    report( NOTICE, kModule ) << "printAmplitudes is being called for a reaction "
                              << "       that is not setup for an amplitude fit."
                              << "       (Nothing more will be printed.)" << endl;

    return;
  }

  AmplitudeManager* ampMan = dynamic_cast< AmplitudeManager* >( intenMan );

  vector<string> ampNames = ampMan->getTermNames();

  // we need to use the AmplitudeManager for this call in order to
  // exercise the GPU code for the amplitude calculation

  AmpVecs aVecs;
  aVecs.loadEvent(kin);
  aVecs.allocateTerms(*intenMan,true);

  ampMan->calcTerms(aVecs);

#ifdef GPU_ACCELERATION
  aVecs.allocateCPUAmpStorage( *intenMan );
  aVecs.m_gpuMan.copyAmpsFromGPU( aVecs );
#endif

  int nAmps = ampNames.size();
  for (unsigned int iamp = 0; iamp < nAmps; iamp++){

    report( INFO, kModule ) << "    ----------------------------------" << endl;
    report( INFO, kModule ) << "      AMPLITUDE = " << ampNames[iamp] << endl;
    report( INFO, kModule ) << "    ----------------------------------" << endl << endl;

    vector< const Amplitude* > ampFactors = ampMan->getFactors(ampNames[iamp]);
    vector <vector <int> > permutations = ampMan->getPermutations(ampNames[iamp]);

    report( INFO, kModule ) << "      PRODUCT OF FACTORS" << endl
                            << "      SUMMED OVER PERMUTATIONS = ( "
                            << aVecs.m_pdAmps[iamp*2] << ", "
                            << aVecs.m_pdAmps[iamp*2+1] << " )" << endl << endl;

    int nPerm = permutations.size();

    if( iamp == nAmps-1 ){

      // for the last amplitude, the pdAmpFactors array will still hold
      // the data for all of the factors and permutations of the amplitude
      // so go ahead and print those to the screen for the user

      for (unsigned int iperm = 0; iperm < nPerm; iperm++){

        report( INFO, kModule ) << "        PERMUTATION = ";
        for (unsigned int ipar = 0; ipar < permutations[iperm].size(); ipar++){
          report( INFO, kModule ) << permutations[iperm][ipar] << " ";
        }

        report( INFO, kModule ) << endl << endl;

        int nFact = ampFactors.size();

        for (unsigned int ifact = 0; ifact < nFact; ifact++){

          report( INFO, kModule ) << "          AMPLITUDE FACTOR = " << ampFactors[ifact]->name() << endl;
          report( INFO, kModule ) << "          IDENTIFIER = " << ampFactors[ifact]->identifier() << endl;
          report( INFO, kModule ) << "          RESULT = ( "
                                  << aVecs.m_pdAmpFactors[ifact*nPerm*2+iperm*2] << ", "
                                  << aVecs.m_pdAmpFactors[ifact*nPerm*2+iperm*2+1] << " )"
                                  << endl << endl;
        }
      }
    }
  }

  // Deallocate memory and return
  report( DEBUG, kModule ) << "Deallocating ampvecs 3:" << endl;
  aVecs.deallocAmpVecs();
}


void
AmpToolsInterface::printIntensity(string reactionName, Kinematics* kin) const {

  IntensityManager* intenMan = intensityManager(reactionName);

  report( INFO, kModule ) << "      ---------------------------------" << endl;
  report( INFO, kModule ) << "        CALCULATING INTENSITY" << endl;
  report( INFO, kModule ) << "      ---------------------------------" << endl << endl;
  double intensity = intenMan->calcIntensity(kin);
  report( INFO, kModule ) << endl << "          INTENSITY = " << intensity << endl << endl << endl;

}


void
AmpToolsInterface::printEventDetails(string reactionName, Kinematics* kin) const {

  printKinematics(reactionName,kin);
  printAmplitudes(reactionName,kin);
  printIntensity(reactionName,kin);
}

void
AmpToolsInterface::forceUserVarRecalculation( bool state ){

  for( unsigned int i = 0; i < m_intensityManagers.size(); ++i ){

    m_intensityManagers[i]->setForceUserVarRecalculation( state );
  }
}

float
AmpToolsInterface::random( float randMax ) const {

  return ( (float) rand() / RAND_MAX ) * randMax;
}

void
AmpToolsInterface::invalidateAmps(){

  report( DEBUG, kModule ) << "Invalidating terms and integrals for all data sets." << endl;

  // it is not necessary to check if the AmpVecs objects
  // are actually used... the vector holds the default
  // objects which are lightweight and have the booleans

  for( int i = 0; i < MAXAMPVECS; ++i ){

    m_ampVecs[i].m_termsValid = false;
    m_ampVecs[i].m_integralValid = false;
  }

  for( map<string,NormIntInterface*>::iterator mapItr = m_normIntMap.begin();
      mapItr != m_normIntMap.end(); ++mapItr ){

    (*mapItr).second->invalidateTerms();
  }

  for( map<string,LikelihoodCalculator*>::iterator mapItr = m_likCalcMap.begin();
      mapItr != m_likCalcMap.end(); ++mapItr ){

    (*mapItr).second->invalidateTerms();
  }
}

void AmpToolsInterface::print_boundPtrCache(){
  MinuitMinimizationManager* man = minuitMinimizationManager();
  std::list<MIObserver*> observers = man->m_observerList;
  report( DEBUG, kModule ) << "There are " << observers.size() << " observers:" << std::endl;
  std::list<MIObserver*>::iterator iter = observers.begin();
  for ( ; iter != observers.end(); ++iter ) { report( DEBUG, kModule ) << *iter << std::endl; }
}
