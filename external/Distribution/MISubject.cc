
// This file is a part of MinuitInterface - a front end for the Minuit minimization
//       package (Minuit itself was authored by Fred James, of CERN)
//
//
// Copyright Cornell University 1993, 1996, All Rights Reserved.
//
// This software written by Lawrence Gibbons, Cornell University.
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
// obtained from Cornell University.
//
// CORNELL MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED.  By way
// of example, but not limitation, CORNELL MAKES NO REPRESENTATIONS OR
// WARRANTIES OF MERCANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT
// THE USE OF THIS SOFTWARE OR DOCUMENTATION WILL NOT INFRINGE ANY PATENTS,
// COPYRIGHTS, TRADEMARKS, OR OTHER RIGHTS.  Cornell University shall not be
// held liable for any liability with respect to any claim by the user or any
// other party arising from use of the program.
//

#include "MinuitInterface/MISubject.h"
#include "MinuitInterface/MIObserver.h"
#include <iostream>

#include "IUAmpTools/report.h"
const char* MISubject::kModule = "MISubject";


MISubject::MISubject() :
   m_observerList()
{}

MISubject::~MISubject()
{}

void
MISubject::attach( MIObserver* anMIObserver ) {

   m_observerList.push_back( anMIObserver );
}

void
MISubject::detach( MIObserver* anMIObserver ) {
   report( DEBUG, kModule ) << "Listing all address of all observers for this object " << this << std::endl;
   report( DEBUG, kModule ) << "There are " << m_observerList.size() << " observers:" << std::endl;
   ObserverList::iterator iter = m_observerList.begin();
   for ( ; iter != m_observerList.end(); ++iter ) { report( DEBUG, kModule ) << *iter << std::endl; }
   report( DEBUG, kModule ) << "Done listing observers" << std::endl;
   report( DEBUG, kModule ) << "Removing observer at address " << anMIObserver << std::endl;
   m_observerList.remove( anMIObserver );
   report( DEBUG, kModule ) << " ++ Done removing observer" << std::endl;
}

void
MISubject::notify() {

   ObserverList::iterator iter = m_observerList.begin();
   for ( ; iter != m_observerList.end(); ++iter ) {
      (*iter)->update( this );
   }
}
