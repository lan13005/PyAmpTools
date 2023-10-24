#ifndef GEN_AMP_H
#define GEN_AMP_H

#include <string>

using namespace std;

class gen_amp
{

public:
    gen_amp(string configfile, string outname);
    ~gen_amp();

    void print_help();
    void generate();

	bool centeredVertex = true;
	bool diag = false;
	bool genFlat = false;
	bool fsRootFormat = false;

	// default upper and lower bounds
	double lowMass = 0.2;
	double highMass = 2.0;

	double beamMaxE   = 12.0;
	double beamPeakE  = 9.0;
	double beamLowE   = 3.0;
	double beamHighE  = 12.0;

	int runNum = 30731;
	unsigned int seed = 0;

	double lowT = 0.0;
	double highT = 12.0;
	double slope = 6.0;

	int nEvents = 10000;
	int batchSize = 10000;

private:
    string configfile;
    string outname;
    string hddmname="";
};

#endif // GEN_AMP_H
