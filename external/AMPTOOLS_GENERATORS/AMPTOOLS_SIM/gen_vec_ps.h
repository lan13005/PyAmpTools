#ifndef GEN_VEC_PS_H
#define GEN_VEC_PS_H

#include <string>

using namespace std;

class gen_vec_ps
{

public:
    gen_vec_ps();
    ~gen_vec_ps();

	void print_help();
	void generate();

	bool diag = false;
	bool genFlat = false;
	bool fsRootFormat = false;

	// default upper and lower bounds
	double lowMass = 1.0; //To take over threshold with a BW omega mass
	double highMass = 2.0;

	double beamMaxE   = 12.0;
	double beamPeakE  = 9.0;
	double beamLowE   = 2.0;
	double beamHighE  = 12.0;

	int runNum = 30731;
	unsigned int seed = 0;

	double lowT = 0.0;
	double highT = 12.0;
	double slope = 6.0;

	int nEvents = 10000;
	int batchSize = 100000;

    string configfile="";
    string outname="";
    string hddmname="";
	string asciiname="";

	vector<string> data_members = {
		"diag",
		"genFlat",
		"fsRootFormat",
		"lowMass",
		"highMass",
		"beamMaxE",
		"beamPeakE",
		"beamLowE",
		"beamHighE",
		"runNum",
		"seed",
		"lowT",
		"highT",
		"slope",
		"nEvents",
		"batchSize",
		"configfile",
		"outname",
		"hddmname"
	};

};

#endif // gen_vec_ps_H
