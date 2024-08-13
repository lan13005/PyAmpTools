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

#include <cassert>

#include "IUAmpTools/AmpVecs.h"
#include "IUAmpTools/IntensityManager.h"
#include "IUAmpTools/DataReader.h"
#include "IUAmpTools/Kinematics.h"

#include "IUAmpTools/report.h"
const char* AmpVecs::kModule = "AmpVecs";

#ifdef GPU_ACCELERATION
#include "GPUManager/GPUManager.h"
#include "cuda_runtime.h"
#endif //GPU_ACCELERATION

AmpVecs::AmpVecs(){
  report( DEBUG, kModule ) << "AmpVecs::AmpVecs() initialization" << endl;

  /// Ints / longs
  m_iNEvents        = 0 ;

  m_iNTrueEvents    = 0 ;
  m_dSumWeights     = 0 ;
  m_iNParticles     = 0 ;
  m_iNTerms         = 0 ;
  m_maxFactPerEvent = 0 ;
  m_userVarsPerEvent = 0;

  // Addresses
  m_pdData      = 0 ;
  m_pdWeights   = 0 ;

  m_pdAmps       = 0 ;
  m_pdAmpFactors = 0 ;
  m_pdUserVars   = 0 ;

  m_pdIntensity      = 0 ;
  m_pdIntegralMatrix = 0 ;

  // Bools
  m_termsValid     = false ;
  m_integralValid  = false ;
  m_dataLoaded     = false;
  m_usesSharedData = false;
  m_sharedDataHost = NULL;

  m_hasNonUnityWeights = false;
  m_hasMixedSignWeights = false;
  m_lastWeightSign = 0;

// #ifdef GPU_ACCELERATION
//   report( DEBUG, kModule ) << "Address of m_gpuMan: " << &m_gpuMan << endl;
// #endif // GPU_ACCELERATION
//   report( DEBUG, kModule ) << "Address of m_iNEvents: " << &m_iNEvents << endl;
//   report( DEBUG, kModule ) << "Address of m_iNTrueEvents: " << &m_iNTrueEvents << endl;
//   report( DEBUG, kModule ) << "Address of m_dSumWeights: " << &m_dSumWeights << endl;
//   report( DEBUG, kModule ) << "Address of m_iNParticles: " << &m_iNParticles << endl;
//   report( DEBUG, kModule ) << "Address of m_iNTerms: " << &m_iNTerms << endl;
//   report( DEBUG, kModule ) << "Address of m_maxFactPerEvent: " << &m_maxFactPerEvent << endl;
//   report( DEBUG, kModule ) << "Address of m_userVarsPerEvent: " << &m_userVarsPerEvent << endl;
//   report( DEBUG, kModule ) << "Address of m_pdData: " << &m_pdData << endl;
//   report( DEBUG, kModule ) << "Address of m_pdWeights: " << &m_pdWeights << endl;
//   report( DEBUG, kModule ) << "Address of m_pdAmps: " << &m_pdAmps << endl;
//   report( DEBUG, kModule ) << "Address of m_pdAmpFactors: " << &m_pdAmpFactors << endl;
//   report( DEBUG, kModule ) << "Address of m_pdUserVars: " << &m_pdUserVars << endl;
//   report( DEBUG, kModule ) << "Address of m_pdIntensity: " << &m_pdIntensity << endl;
//   report( DEBUG, kModule ) << "Address of m_pdIntegralMatrix: " << &m_pdIntegralMatrix << endl;
//   report( DEBUG, kModule ) << "Address of m_termsValid: " << &m_termsValid << endl;
//   report( DEBUG, kModule ) << "Address of m_integralValid: " << &m_integralValid << endl;
//   report( DEBUG, kModule ) << "Address of m_dataLoaded: " << &m_dataLoaded << endl;
//   report( DEBUG, kModule ) << "Address of m_usesSharedData: " << &m_usesSharedData << endl;
//   report( DEBUG, kModule ) << "Address of m_hasNonUnityWeights: " << &m_hasNonUnityWeights << endl;
//   report( DEBUG, kModule ) << "Address of m_hasMixedSignWeights: " << &m_hasMixedSignWeights << endl;
//   report( DEBUG, kModule ) << "Address of m_lastWeightSign: " << &m_lastWeightSign << endl;

  report( DEBUG, kModule ) << "AmpVecs::AmpVecs() initialization completed" << endl;
}

void
AmpVecs::deallocAmpVecs()
{
  report( DEBUG, kModule ) << "AmpVecs::deallocAmpVecs() destructor" << endl;
  m_iNEvents        = 0 ;
  m_iNTrueEvents    = 0 ;
  m_dSumWeights     = 0 ;
  m_iNParticles     = 0 ;
  m_iNTerms         = 0 ;
  m_maxFactPerEvent = 0 ;
  m_userVarsPerEvent = 0;

  m_termsValid    = false ;
  m_integralValid = false ;
  m_dataLoaded    = false ;

  m_hasNonUnityWeights = false;
  m_hasMixedSignWeights = false;
  m_lastWeightSign = 0;

  // this handles the case that
  // other classes may be looking
  // to this class for the data
  clearFourVecs();

  if(m_pdWeights){
    delete[] m_pdWeights;
    report( DEBUG, kModule ) << "AmpVecs::deallocAmpVecs() deleting m_pdWeights" << endl;
  }
  m_pdWeights=0;

  if(m_pdIntensity){
    delete[] m_pdIntensity;
    report( DEBUG, kModule ) << "AmpVecs::deallocAmpVecs() deleting m_pdIntensity" << endl;
  }
  m_pdIntensity=0;

  if(m_pdIntegralMatrix){
    delete[] m_pdIntegralMatrix;
    report( DEBUG, kModule ) << "AmpVecs::deallocAmpVecs() deleting m_pdIntegralMatrix" << endl;
  }
  m_pdIntegralMatrix=0;

  if(m_pdUserVars){
    delete[] m_pdUserVars;
    report( DEBUG, kModule ) << "AmpVecs::deallocAmpVecs() deleting m_pdUserVars" << endl;
  }
  m_pdUserVars=0;

  m_userVarsOffset.clear();

#ifndef GPU_ACCELERATION
  report( DEBUG, kModule ) << "AmpVecs::deallocAmpVecs() CPU deleting m_pdAmps and m_pdAmpFactors" << endl;
  if(m_pdAmps){
    delete[] m_pdAmps;
    report( DEBUG, kModule ) << "AmpVecs::deallocAmpVecs() CPU deleting m_pdAmps" << endl;
  }
  m_pdAmps=0;

  if(m_pdAmpFactors){
    delete[] m_pdAmpFactors;
    report( DEBUG, kModule ) << "AmpVecs::deallocAmpVecs() CPU deleting m_pdAmpFactors" << endl;
  }
  m_pdAmpFactors=0;

#else
  report( DEBUG, kModule ) << "AmpVecs::deallocAmpVecs() GPU deleting m_pdAmps and m_pdAmpFactors" << endl;
  //Deallocate "pinned memory"
  if(m_pdAmps){
    cudaFreeHost(m_pdAmps);
    report( DEBUG, kModule ) << "AmpVecs::deallocAmpVecs() GPU deleting m_pdAmps" << endl;
  }
  m_pdAmps=0;

  if(m_pdAmpFactors){
    cudaFreeHost(m_pdAmpFactors);
    report( DEBUG, kModule ) << "AmpVecs::deallocAmpVecs() GPU deleting m_pdAmpFactors" << endl;
  }
  m_pdAmpFactors=0;

  m_gpuMan.clearAll();

#endif // GPU_ACCELERATION
}

void
AmpVecs::clearFourVecs(){
  report( DEBUG, kModule ) << "AmpVecs::clearFourVecs() clearing four-vectors called" << endl;

  if( m_usesSharedData ){
    report ( DEBUG, kModule ) << "AmpVecs::clearFourVecs() called on shared data" << endl;
    // this should always be true because an
    // object using shared data shouldn't share it
    // again with another object -- only the host
    // should do the sharing
    assert( m_sharedDataFriends.empty() );

    m_sharedDataHost->removeFriend( this );
    m_sharedDataHost = NULL;
    m_usesSharedData = false;

    // set the pointer to zero to avoid deleting data
    // that others need in the steps below
    m_pdData = 0;
  }

  if( !m_sharedDataFriends.empty() ) {

    report( DEBUG, kModule ) << "AmpVecs::clearFourVecs() m_sharedDataFriends was not empty" << endl;

    // transfer the ownership of the data
    // to the first member of the set of
    // objects that are sharing this data

    std::set<AmpVecs*>::iterator avItr = m_sharedDataFriends.begin();

    AmpVecs* newDataOwner = *avItr;
    report( DEBUG, kModule ) << "AmpVecs::clearFourVecs() erasing shared data" << endl;
    m_sharedDataFriends.erase( avItr );

    newDataOwner->claimDataOwnership( m_sharedDataFriends );

    // clear out the shared data friends since the new
    // owner is now responsible for them
    m_sharedDataFriends.clear();

    // set the pointer to zero to avoid deleting data
    // that others need in the steps below
    m_pdData = 0;
  }

  // proceed as normal by flushing the four-vectors
  if(m_pdData)
    delete[] m_pdData;

  m_pdData=0;
}

void
AmpVecs::loadEvent( const Kinematics* pKinematics, unsigned long long iEvent,
                    unsigned long long iNTrueEvents ){
  // allocate memory and set variables
  //  if this is the first call to this method

  if (m_pdData == NULL){
    // if every other 2000 events, print a message
    if( iEvent % 2500 == 0 )
      report( DEBUG, kModule ) << "AmpVecs::loadEvent() allocating memory for data" << endl;

    m_iNTrueEvents = iNTrueEvents;
    m_iNEvents = iNTrueEvents;

#ifdef GPU_ACCELERATION
    m_iNEvents = GPUManager::calcNEventsGPU(iNTrueEvents);
#endif

    m_iNParticles = pKinematics->particleList().size();
    assert(m_iNParticles);

    m_pdData = new GDouble[4*m_iNParticles*m_iNEvents];
    m_pdWeights = new GDouble[m_iNEvents];
    // report( DEBUG, kModule ) << "AmpVecs::loadEvent() allocated memory for data" << endl;
  }

  // check to be sure we won't exceed the bounds of the array
  // report( DEBUG, kModule ) << "AmpVecs::loadEvent() loading event " << iEvent << endl;
  assert( iEvent < m_iNEvents );

  // report( DEBUG, kModule ) << "AmpVecs::loadEvent() loading event " << iEvent << endl;
  for (int iParticle = 0; iParticle < m_iNParticles; iParticle++){
    m_pdData[4*iEvent*m_iNParticles+4*iParticle+0]=pKinematics->particle(iParticle).E();
    m_pdData[4*iEvent*m_iNParticles+4*iParticle+1]=pKinematics->particle(iParticle).Px();
    m_pdData[4*iEvent*m_iNParticles+4*iParticle+2]=pKinematics->particle(iParticle).Py();
    m_pdData[4*iEvent*m_iNParticles+4*iParticle+3]=pKinematics->particle(iParticle).Pz();
  }

  m_pdWeights[iEvent] = pKinematics->weight();

  m_termsValid = false;
  m_integralValid = false;
  m_dataLoaded = true;
  m_userVarsOffset.clear();
  // report( DEBUG, kModule ) << "AmpVecs::loadEvent() completed" << endl;
}


void
AmpVecs::loadData( DataReader* pDataReader ){
  report( DEBUG, kModule ) << "AmpVecs::loadData() loading data" << endl;
  //  Make sure no data is already loaded

  if( m_pdData!=0 || m_pdWeights!=0 ){
    report( ERROR, kModule ) << "Trying to load data into a non-empty AmpVecs object\n"<<flush;
    assert(false);
  }

  // Get the number of events and reset the data reader

  report( DEBUG, kModule ) << "AmpVecs::loadData() resetting: " << pDataReader->identifier() << endl;
  pDataReader->resetSource();
  m_iNTrueEvents = pDataReader->numEvents();
  report( DEBUG, kModule ) << "AmpVecs::loadData() numEvents: " << m_iNTrueEvents << endl;

  // try to print an informative message -- this can be normal behavior in
  // an MPI job with a sparse background source and many concurrent processess
  if( m_iNTrueEvents < 1 ){
    report( NOTICE, kModule ) << "DataReader " << pDataReader->identifier() << " does not contain any events." << endl;
    string readerName = pDataReader->name();
    vector< string > readerArgs = pDataReader->arguments();

    report( NOTICE, kModule ) << readerName << " with arguments:  \n\t";
    for( std::vector<string>::iterator arg = readerArgs.begin(); arg != readerArgs.end(); ++arg ){

      report( NOTICE, kModule ) << *arg << "\t";
    }
    report( NOTICE, kModule ) << "\n does not contain any events." << endl;
  }

  // Loop over events and load each one individually

  Kinematics* pKinematics;
  for(unsigned long long iEvent = 0; iEvent < m_iNTrueEvents; iEvent++){
    // report( DEBUG, kModule ) << "AmpVecs::loadData() loading event " << iEvent << endl;
    pKinematics = pDataReader->getEvent();
    // report( DEBUG, kModule ) << "AmpVecs::loadData() got event " << iEvent << endl;
    loadEvent(pKinematics, iEvent, m_iNTrueEvents );
    // report( DEBUG, kModule ) << "AmpVecs::loadData() loaded event " << iEvent << endl;

    float weight = pKinematics->weight();

    // fill some booleans that contain collective information about the weights
    if( weight != 1 ) m_hasNonUnityWeights = true;
    if( m_lastWeightSign == 0 ) m_lastWeightSign = weight;
    int thisWeightSign = ( weight > 0 ? 1 : 0 );
    thisWeightSign = ( weight < 0 ? -1 : thisWeightSign );
    if( thisWeightSign * m_lastWeightSign < 0 ) m_hasMixedSignWeights = true;
    m_lastWeightSign = thisWeightSign;

    m_dSumWeights += pKinematics->weight();
    if (iEvent < (m_iNTrueEvents - 1)) delete pKinematics;
  }

  // Fill any remaining space in the data array with the last event's kinematics

  for (unsigned long long iEvent = m_iNTrueEvents; iEvent < m_iNEvents; iEvent++){
    loadEvent(pKinematics, iEvent, m_iNTrueEvents );
  }

  if( m_iNTrueEvents )
  	delete pKinematics;

  m_termsValid = false;
  m_integralValid = false;
  m_dataLoaded = true;
  m_userVarsOffset.clear();
}


void
AmpVecs::allocateTerms( const IntensityManager& intenMan, bool bAllocIntensity ){
  report( DEBUG, kModule ) << "AmpVecs::allocateTerms() allocating terms" << endl;
  m_iNTerms           = intenMan.getTermNames().size();
  m_termNames         = intenMan.getTermNames();
  m_maxFactPerEvent   = intenMan.maxFactorStoragePerEvent();
  m_userVarsPerEvent  = intenMan.userVarsPerEvent();

  if( m_pdAmps!=0 || m_pdAmpFactors!=0 || m_pdUserVars!=0 || m_pdIntensity!=0 )
  {
    report( ERROR, kModule ) << "ERROR:  trying to reallocate terms in AmpVecs after\n" << flush;
    report( ERROR, kModule ) << "        they have already been allocated.  Please\n" << flush;
    report( ERROR, kModule ) << "        deallocate them first." << endl;
    assert(false);
  }

  if ( !m_dataLoaded ){
    report( ERROR, kModule ) << "ERROR: trying to allocate space for terms in\n" << flush;
    report( ERROR, kModule ) << "       AmpVecs before any events have been loaded\n" << flush;
    assert(false);
  }

  m_pdIntegralMatrix = new double[2*m_iNTerms*m_iNTerms];

  //Allocate the Intensity only when needed
  if( bAllocIntensity )
  {
    m_pdIntensity = new GDouble[m_iNEvents];
  }

  if( m_userVarsPerEvent > 0 ){

    // if there is no user data, we need pdUserVars to be NULL
    // in order to ensure backwards compatibility with older
    // amplitude definitions

    m_pdUserVars = new GDouble[m_iNEvents * m_userVarsPerEvent];
  }

#ifndef GPU_ACCELERATION

  m_pdAmps = new GDouble[m_iNEvents * intenMan.termStoragePerEvent()];
  m_pdAmpFactors = new GDouble[m_iNEvents * m_maxFactPerEvent];

#else

  m_gpuMan.init( *this, !intenMan.needsUserVarsOnly() );
  m_gpuMan.copyDataToGPU( *this, !intenMan.needsUserVarsOnly() );

#endif // GPU_ACCELERATION

  m_termsValid    = false;
  m_integralValid = false;
}

#ifdef GPU_ACCELERATION
void
AmpVecs::allocateCPUAmpStorage( const IntensityManager& intenMan ){
  report( DEBUG, kModule ) << "AmpVecs::allocateCPUAmpStorage() allocating CPU storage for amplitudes" << endl;
  // we should start with unallocated memory
  assert( m_pdAmps == NULL && m_pdAmpFactors == NULL );

  // allocate as "pinned memory" for fast CPU<->GPU memcopies
  cudaMallocHost( (void**)&m_pdAmps, m_iNEvents * intenMan.termStoragePerEvent() * sizeof(GDouble) );
  cudaMallocHost( (void**)&m_pdAmpFactors, m_iNEvents * m_maxFactPerEvent * sizeof(GDouble));

  cudaError_t cudaErr = cudaGetLastError();
  if( cudaErr != cudaSuccess  ){

    report( ERROR, kModule ) << "CUDA memory allocation error: "<< cudaGetErrorString( cudaErr ) << endl;
    assert( false );
  }
}
#endif

Kinematics*
AmpVecs::getEvent( int iEvent ){

  // check to be sure the event request is realistic
  assert( iEvent < m_iNTrueEvents );

  vector< TLorentzVector > particleList;

  for( int iPart = 0; iPart < m_iNParticles; ++iPart ){

    int i = iEvent*4*m_iNParticles + 4*iPart;
    particleList.push_back( TLorentzVector( m_pdData[i+1], m_pdData[i+2],
                                            m_pdData[i+3], m_pdData[i] ) );
  }

  return new Kinematics( particleList, m_pdWeights[iEvent] );
}

void
AmpVecs::shareDataWith( AmpVecs* targetAmpVecs ){
  report( DEBUG, kModule ) << "AmpVecs::shareDataWith() sharing data with " << targetAmpVecs << endl;

  // if this object is already using shared data
  // then it should rely on the host to distribute
  // the data to the new object in order to keep
  // the set of shared data friends complete
  if( m_usesSharedData ){
    
    m_sharedDataHost->shareDataWith( targetAmpVecs );
    return;
  }
  
  // the algorithm assumes that the target AmpVecs
  // object is empty to begin with -- if this is not
  // the case there is programming error somewhere
  assert( targetAmpVecs->m_pdData == 0 );

  targetAmpVecs->m_iNEvents = m_iNEvents;
  targetAmpVecs->m_iNTrueEvents = m_iNTrueEvents;
  targetAmpVecs->m_dSumWeights = m_dSumWeights;
  targetAmpVecs->m_iNParticles = m_iNParticles;

  targetAmpVecs->m_pdData = m_pdData;

  // while we are really going to share the four-vectors,
  // we will give the target a copy of the weights -- this
  // avoids complications if the four-vectors can later
  // be flusehd from memory but the weights need to remain
  // for the fit

  targetAmpVecs->m_pdWeights = new GDouble[m_iNEvents];
  memcpy( targetAmpVecs->m_pdWeights, m_pdWeights,
	  sizeof(GDouble)*m_iNEvents );

  targetAmpVecs->m_dataLoaded = true;
  targetAmpVecs->m_hasNonUnityWeights = m_hasNonUnityWeights;
  targetAmpVecs->m_hasMixedSignWeights = m_hasMixedSignWeights;

  m_sharedDataFriends.insert( targetAmpVecs );
  targetAmpVecs->m_usesSharedData = true;
  targetAmpVecs->m_sharedDataHost = this; 
}

void
AmpVecs::removeFriend( AmpVecs* dataFriend ){

  m_sharedDataFriends.erase( dataFriend );
}

void
AmpVecs::claimDataOwnership( set< AmpVecs* > sharedFriends ){

  m_usesSharedData = false;
  m_sharedDataHost = NULL;
  m_sharedDataFriends = sharedFriends;

  // need to tell the friends that this class is the new host
  for( set< AmpVecs* >::iterator avItr = sharedFriends.begin();
       avItr != sharedFriends.end(); ++avItr ){

    (*avItr)->m_sharedDataHost = this;
  }
}
