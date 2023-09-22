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
// #include "IUAmpToolsMPI/AmpToolsInterfaceMPI.h"
// #include "IUAmpToolsMPI/DataReaderMPI.h"
#include "IUAmpTools/AmpToolsInterface.h"
#include "IUAmpTools/DataReader.h"
#include "IUAmpTools/FitResults.h"
#include "IUAmpTools/ConfigFileParser.h"
#include "IUAmpTools/ConfigurationInfo.h"

using std::complex;
using namespace std;

// int rank_mpi;
int size;

double likelihood(AmpToolsInterface* ati, ParameterManager* parMgr, vector< vector<string> > pois) {
    double nll = 1e7;  
//    if(rank_mpi==0) {
		nll = ati->likelihood();
        cout << "LIKELIHOOD BEFORE SETTING PARAMETERS:  " << nll << endl;      
//        parMgr = ati.parameterManager();
//        ati.reinitializePars();
//        parMgr->setAmpParameter( poi, value );
//        nll = ati.likelihood();
     }

    return nll;
}

int main( int argc, char* argv[] ){
	// MPI_Init( &argc, &argv );
	
	// MPI_Comm_rank( MPI_COMM_WORLD, &rank_mpi );
	// MPI_Comm_size( MPI_COMM_WORLD, &size );

	string configfile;

	for (int i = 1; i < argc; i++){
		string arg(argv[i]);	
		if (arg == "-c"){  
			if ((i+1 == argc) || (argv[i+1][0] == '-')) arg = "-h";
			else  configfile = argv[++i]; }
		if (arg == "-h"){
			cout << endl << " Usage for: " << argv[0] << endl << endl;
			cout << "   -c <file>\t\t\t\t config file" << endl;
			exit(1);}
	}

	if (configfile==""){
		cout << "No config file specified.  Exiting..." << endl << endl;
		exit(1);
	}
	cout << "Config file location: " << configfile << endl;
	
	ConfigFileParser parser(configfile);
	ConfigurationInfo* cfgInfo = parser.getConfigurationInfo();
	// if( rank_mpi == 0 ) cfgInfo->display();
	
	AmpToolsInterface::registerAmplitude( Zlm() );
	// AmpToolsInterface::registerDataReader( DataReaderMPI<ROOTDataReader>() );
	AmpToolsInterface::registerDataReader( ROOTDataReader() );
	
	// AmpToolsInterfaceMPI ati( cfgInfo );
	AmpToolsInterface ati( cfgInfo );
	ParameterManager* parMgr = ati.parameterManager();
	vector< vector<string> > pois; 
	likelihood(&ati, parMgr, pois);
	
	// ati.exitMPI();
    // MPI_Finalize();
	return 0;
}

