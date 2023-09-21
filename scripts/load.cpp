#include <iostream>
#include <fstream>
#include <sstream>
#include <complex>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <limits>

#include "TSystem.h"

#include "AMPTOOLS_AMPS_DATAIO/ROOTDataReader.h"
#include "AMPTOOLS_AMPS_DATAIO/Zlm.h"

#include "MinuitInterface/MinuitMinimizationManager.h"
#include "IUAmpToolsMPI/AmpToolsInterfaceMPI.h"
#include "IUAmpToolsMPI/DataReaderMPI.h"
#include "IUAmpTools/FitResults.h"
#include "IUAmpTools/ConfigFileParser.h"
#include "IUAmpTools/ConfigurationInfo.h"

using std::complex;
using namespace std;

int rank_mpi;
int size;

void likelihood(AmpToolsInterfaceMPI* ati, ParameterManager* parMgr, vector< vector<string> > pois) {
    double curLH = 1e7;  
    if(rank_mpi==0) {
        cout << "LIKELIHOOD BEFORE SETTING PARAMETERS:  " << ati.likelihood() << endl;      
//        parMgr = ati.parameterManager();
//        ati.reinitializePars();
//        parMgr->setAmpParameter( poi, value );
//        curLH = ati.likelihood();
     }
    ati.exitMPI();
    MPI_Finalize();

    return curLH;
}

int main( int argc, char* argv[] ){
	MPI_Init( &argc, &argv );
	
	MPI_Comm_rank( MPI_COMM_WORLD, &rank_mpi );
	MPI_Comm_size( MPI_COMM_WORLD, &size );
	
	ConfigFileParser parser(configfile);
	ConfigurationInfo* cfgInfo = parser.getConfigurationInfo();
	if( rank_mpi == 0 ) cfgInfo->display();
	
	AmpToolsInterface::registerAmplitude( Zlm() );
	AmpToolsInterface::registerDataReader( DataReaderMPI<ROOTDataReader>() );
	
	AmpToolsInterfaceMPI ati( cfgInfo );
	ParameterManager* parMgr = ati.parameterManager();
	vector< vector<string> > pois; 
	likelihood(ati, parMgr, pois);
	
	return 0;
}

