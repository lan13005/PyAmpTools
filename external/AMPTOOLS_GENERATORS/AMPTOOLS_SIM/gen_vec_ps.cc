//Generator for generic vector-pseudoscalar
//Based on b1(1235)->omega pi by A. M. Foda https://github.com/amfodajlab and gen_amp by Alex Austregesilo https://github.com/aaust
#include <iostream>
#include <fstream>
#include <complex>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <cassert>
#include <cstdlib>

#include "particleType.h"

#include "AMPTOOLS_DATAIO/DataWriter.h"
#include "AMPTOOLS_DATAIO/ROOTDataWriter.h"
#include "AMPTOOLS_DATAIO/FSRootDataWriter.h"

#ifdef HDDM
#include "AMPTOOLS_MCGEN/HDDMDataWriter.h"
#endif

#include "AMPTOOLS_AMPS/omegapiAngles.h"
#include "AMPTOOLS_AMPS/Vec_ps_refl.h"
#include "AMPTOOLS_AMPS/Piecewise.h"
#include "AMPTOOLS_AMPS/BreitWigner.h"
#include "AMPTOOLS_AMPS/Uniform.h"
#include "AMPTOOLS_AMPS/OmegaDalitz.h"
#include "AMPTOOLS_AMPS/PhaseOffset.h"
#include "AMPTOOLS_AMPS/ComplexCoeff.h"

#include "AMPTOOLS_MCGEN/ProductionMechanism.h"
#include "AMPTOOLS_MCGEN/GammaPToNPartP.h"
#include "AMPTOOLS_MCGEN/NBodyPhaseSpaceFactory.h"

#include "IUAmpTools/AmpToolsInterface.h"
#include "IUAmpTools/ConfigFileParser.h"
#include "IUAmpTools/PlotGenerator.h"
#include "IUAmpTools/FitResults.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TLorentzVector.h"
#include "TLorentzRotation.h"
#include "TRandom3.h"

#include "UTILITIES/CobremsGeneration.h"
#include "UTILITIES/BeamProperties.h"

#include "AMPTOOLS_SIM/gen_vec_ps.h"

using std::complex;
using namespace std;

gen_vec_ps::gen_vec_ps()
{
}

gen_vec_ps::~gen_vec_ps()
{
}

void
gen_vec_ps::print_help(){
    cout << endl << " GEN_VEC_PS: Simulation attributes:" << endl << endl;
	cout << "----- General -----" << endl;
	cout << "configfile:    Configuration file" << endl;
	cout << "outname:       Output file name" << endl;
	cout << "nEvents:       Number of events to generate" << endl;
	cout << "----- Kinematics -----" << endl;
	cout << "genFlat:       Generate flat in M(X) (no physics)" << endl;
	cout << "lowMass:       Lower edge of mass range (GeV)" << endl;
	cout << "highMass:      Upper edge of mass range (GeV)" << endl;
	cout << "beamMaxE:      Electron beam energy (or photon energy endpoint)" << endl;
	cout << "beamPeakE:     Coherent peak photon energy" << endl;
	cout << "beamLowE:      Minimum photon energy to simulate events" << endl;
	cout << "beamHighE:     Maximum photon energy to simulate events" << endl;
	cout << "slope:         Momentum transfer slope" << endl;
	cout << "lowT:          Minimum momentum transfer" << endl;
	cout << "highT:         Maximum momentum transfer" << endl;
	cout << "----- Output and Conditioning -----" << endl;
	cout << "runNum:        Run number assigned to generated events" << endl;
	cout << "seed:          Random number seed initialization" << endl;
	cout << "diag:          Plot only diagnostic histograms" << endl;
	cout << "fsRootFormat:  Enable output in FSRoot format" << endl;
}

void
gen_vec_ps::generate(){

	if( configfile.size() == 0 || outname.size() == 0 ){
		cout << "No config file or output specificed:  run gen_vec_ps -h for help" << endl;
		exit(1);
	}

	// open config file and be sure only one reaction is specified
	ConfigFileParser parser( configfile );
	ConfigurationInfo* cfgInfo = parser.getConfigurationInfo();
	assert( cfgInfo->reactionList().size() == 1 );
	ReactionInfo* reaction = cfgInfo->reactionList()[0];

	// check for unstable particle at lower vertex
	vector<Particle_t> ParticlesLowerVertex;
	vector<double> massesLowerVertex;
	double thresholdLowerVertex = 0;
	vector< BreitWignerGenerator > bwGenLowerVertex;
	vector< vector<string> > lowerVertexKeywords = cfgInfo->userKeywordArguments("lowerVertex");
	if(lowerVertexKeywords.size() == 1) {
	  vector<string> keywordArgs = lowerVertexKeywords[0];
	  bwGenLowerVertex.push_back( BreitWignerGenerator( atof(keywordArgs[0].c_str()), atof(keywordArgs[1].c_str())) );
	  cout << "Unstable particle at lower vertex: mass = " << keywordArgs[0] << "GeV , width = " << keywordArgs[1] << "GeV" << endl;
	  for(unsigned int j=2; j<keywordArgs.size(); j++) {
	    ParticlesLowerVertex.push_back(ParticleEnum(keywordArgs[j].c_str()));
	    massesLowerVertex.push_back(ParticleMass(ParticlesLowerVertex[j-2]));
	    thresholdLowerVertex += ParticleMass(ParticlesLowerVertex[j-2]);
	  }
	}
	else if(lowerVertexKeywords.size() > 1) {
		cout<<"Multiple unstable particles at lower vertex provided"<<endl;
		exit(1);
	}

	// use particletype.h to convert reaction particle names (for upper vertex)
	vector<Particle_t> Particles;
	// don't include non-nucleon lower vertex decay particles in meson decay
	unsigned int maxUpperVertexChild = reaction->particleList().size();
	if(bwGenLowerVertex.size() == 1) maxUpperVertexChild -= (ParticlesLowerVertex.size()-1);
	for (unsigned int i = 0; i < maxUpperVertexChild; i++){
		TString particleName = reaction->particleList()[i].c_str();
		particleName.ReplaceAll("1","");  particleName.ReplaceAll("2",""); // ignore distinguishable particle notation
		Particle_t locEnum = ParticleEnum(particleName.Data());
		// Beam particle is always photon
		if (locEnum == 0 && i > 0)
			cout << "ConfigFileParser WARNING:  unknown particle type \"" << particleName.Data() << "\"" << endl;
		Particles.push_back(ParticleEnum(particleName.Data()));
    }

	// set vector masses from particle list
	vector<double> vectorMasses;
	vector<double> childMasses;
	for (unsigned int i=2; i<Particles.size(); i++) {
		if(i==2) childMasses.push_back(ParticleMass(Particles[i]));
		else vectorMasses.push_back(ParticleMass(Particles[i]));
	}

	// loop to look for resonance in config file
	// currently only one at a time is supported
	const vector<ConfigFileLine> configFileLines = parser.getConfigFileLines();
	double resonance[]={1.235, 1.0};
	bool foundResonance = false;
	for (vector<ConfigFileLine>::const_iterator it=configFileLines.begin(); it!=configFileLines.end(); it++) {
	  if ((*it).keyword() == "define") {
		if ((*it).arguments()[0] == "b1" || (*it).arguments()[0] == "K1"){
	      if ( (*it).arguments().size() != 3 )
		continue;
	      resonance[0]=atof((*it).arguments()[1].c_str());
	      resonance[1]=atof((*it).arguments()[2].c_str());
	      cout << "Distribution seeded with resonance " << (*it).arguments()[0] << " : mass = " << resonance[0] << "GeV , width = " << resonance[1] << "GeV" << endl;
	      foundResonance = true;
	      break;
	    }
	  }
	}
	if (!foundResonance)
		cout << "ConfigFileParser WARNING:  no known resonance found, seed with flat mass distribution" << endl;

	// find vector parameters from config file
	double vecMass = 0;
	double vecWidth = 0;
	double threshold = childMasses[0];
	bool foundVector = false;
	for (vector<ConfigFileLine>::const_iterator it=configFileLines.begin(); it!=configFileLines.end(); it++) {
		if ((*it).keyword() == "define" && (*it).arguments()[0] == "vector" && (*it).arguments().size() == 3) {
			vecMass = atof((*it).arguments()[1].c_str());
			vecWidth = atof((*it).arguments()[2].c_str());
			childMasses.push_back(vecMass);
			threshold += vecMass;
			foundVector = true;
		}
	}
	if (!foundVector) {
		cout << "ConfigFileParser ERROR: no vector found, cannot continue" << endl;
		return;
	}

	// random number initialization (set to 0 by default)
	TRandom3* gRandom = new TRandom3();
	gRandom->SetSeed(seed);
	seed = gRandom->GetSeed();
	cout << "TRandom3 Seed : " << seed << endl;

	// setup AmpToolsInterface
	AmpToolsInterface::registerAmplitude( Vec_ps_refl() );
	AmpToolsInterface::registerAmplitude( BreitWigner() );
	AmpToolsInterface::registerAmplitude( Uniform() );
	AmpToolsInterface::registerAmplitude( OmegaDalitz() );
	AmpToolsInterface::registerAmplitude( Piecewise() );
	AmpToolsInterface::registerAmplitude( PhaseOffset() );
	AmpToolsInterface::registerAmplitude( ComplexCoeff() );

	AmpToolsInterface ati( cfgInfo, AmpToolsInterface::kMCGeneration );

	double polAngle = -1;//amorphous
	// loop to look for beam configuration file
	TString beamConfigFile;
	const vector<ConfigFileLine> configFileLinesBeam = parser.getConfigFileLines();
	for (vector<ConfigFileLine>::const_iterator it=configFileLinesBeam.begin(); it!=configFileLinesBeam.end(); it++) {
		if ((*it).keyword() == "define") {
			TString beamArgument =  (*it).arguments()[0].c_str();
			if(beamArgument.Contains("beamconfig")) {
				beamConfigFile = (*it).arguments()[1].c_str();
				BeamProperties beamProp(beamConfigFile);
				polAngle = beamProp.GetPolAngle();
			}
		}
	}
	if(beamConfigFile.Length() == 0) {
		cout<<"WARNING: Couldn't find beam configuration file -- write local version"<<endl;

		beamConfigFile = "local_beam.conf";
		ofstream locBeamConfigFile;
		locBeamConfigFile.open(beamConfigFile.Data());
		locBeamConfigFile<<"ElectronBeamEnergy "<<beamMaxE<<endl;       // electron beam energy
		locBeamConfigFile<<"CoherentPeakEnergy "<<beamPeakE<<endl;      // coherent peak energy
		locBeamConfigFile<<"PhotonBeamLowEnergy "<<beamLowE<<endl;      // photon beam low energy
		locBeamConfigFile<<"PhotonBeamHighEnergy "<<beamHighE<<endl;    // photon beam high energy
		locBeamConfigFile.close();
	}

	ProductionMechanism::Type type =
		( (genFlat || !foundResonance) ? ProductionMechanism::kFlat : ProductionMechanism::kResonant );

	// generate over a range of mass
	// start with threshold or lowMass, whichever is higher
	GammaPToNPartP resProd = GammaPToNPartP( threshold<lowMass ? lowMass : threshold, highMass, childMasses, ProductionMechanism::kProton, type, slope, lowT, highT, seed, beamConfigFile );

	vector< BreitWignerGenerator > m_bwGen;
	m_bwGen.push_back( BreitWignerGenerator(vecMass, vecWidth) );

	// seed the distribution with a sum of noninterfering Breit-Wigners
	// we can easily compute the PDF for this and divide by that when
	// doing accept/reject -- improves efficiency if seeds are picked well

	if( !genFlat && foundResonance){
		// the lines below should be tailored by the user for the particular desired
		// set of amplitudes -- doing so will improve efficiency.  Leaving as is
		// won't make MC incorrect, it just won't be as fast as it could be

		resProd.addResonance( resonance[0], resonance[1],  1.0 );
	}

	vector< int > pTypes;
	for (unsigned int i=0; i<Particles.size(); i++)
	    pTypes.push_back( Particles[i] );
	for (unsigned int i=1; i<ParticlesLowerVertex.size(); i++)
        pTypes.push_back( ParticlesLowerVertex[i] );

    #ifdef HDDM
	HDDMDataWriter* hddmOut = NULL;
	if( hddmname.size() != 0 ) hddmOut = new HDDMDataWriter( hddmname, runNum, seed);
    #endif

	DataWriter* rootOut = ( fsRootFormat ?
				static_cast< DataWriter*>( new FSRootDataWriter( reaction->particleList().size()-1, outname ) ) :
				static_cast< DataWriter* >( new ROOTDataWriter( outname ) ) );

	TFile* diagOut = new TFile( "gen_vec_ps_diagnostic.root", "recreate" );
	ostringstream locStream;
	ostringstream locIsobarStream;
	ostringstream locIsobar2Stream;
	for (unsigned int i=2; i<Particles.size(); i++){
	  locStream << ParticleName_ROOT(Particles[i]);
	  if ( i> 2 )
	    locIsobarStream << ParticleName_ROOT(Particles[i]);
	}
	string locHistTitle = string("Resonance Mass ;") + locStream.str() + string(" Invariant Mass (GeV/c^{2});");
	string locIsobarTitle = string("Isobar Mass ;") + locIsobarStream.str() + string(" Invariant Mass (GeV/c^{2});");
	string locIsobar2Title = string("Isobar2 Mass ;") + locIsobar2Stream.str() + string(" Invariant Mass (GeV/c^{2});");

	TH1F* mass = new TH1F( "M", locHistTitle.c_str(), 180, lowMass, highMass );
	TH1F* massW = new TH1F( "M_W", ("Weighted "+locHistTitle).c_str(), 180, lowMass, highMass );
	massW->Sumw2();
	TH1F* intenW = new TH1F( "intenW", "True PDF / Gen. PDF", 1000, 0, 100 );
	TH2F* intenWVsM = new TH2F( "intenWVsM", "Ratio vs. M", 100, lowMass, highMass, 1000, 0, 10 );

	TH1F* t = new TH1F( "t", "-t Distribution", 200, 0, 2 );

	TH1F* M_isobar = new TH1F( "M_isobar", locIsobarTitle.c_str(), 200, 0, 2 );
	TH1F* M_isobar2 = new TH1F( "M_isobar2", locIsobar2Title.c_str(), 200, 0, 2 );
	TH1F* M_recoil = new TH1F( "M_recoil", "; Recoil mass (GeV)", 200, 0, 2 );
	TH1F* M_recoilW = new TH1F( "M_recoilW", "; Weighted Recoil mass (GeV)", 200, 0, 2 );
	TH1F* M_p1 = new TH1F( "M_p1", "p1", 200, 0, 2 );
	TH1F* M_p2 = new TH1F( "M_p2", "p2", 200, 0, 2 );
	TH1F* M_p3 = new TH1F( "M_p3", "p3", 200, 0, 2 );
	TH1F* M_p4 = new TH1F( "M_p4", "p4", 200, 0, 2 );

    TH2F* M_dalitz = new TH2F( "M_dalitz", "dalitzxy", 200, -2, 2, 200, -2, 2);
	TH1F* lambda = new TH1F( "lambda", "#lambda_{#omega}", 120, 0.0, 1.2);

	TH2F* CosTheta_psi = new TH2F( "CosTheta_psi", "cos#theta vs. #psi", 180, -3.14, 3.14, 100, -1, 1);
	TH2F* M_CosTheta = new TH2F( "M_CosTheta", "M vs. cos#vartheta", 180, lowMass, highMass, 200, -1, 1);
	TH2F* M_Phi = new TH2F( "M_Phi", "M vs. #varphi", 180, lowMass, highMass, 200, -3.14, 3.14);
	TH2F* M_CosThetaH = new TH2F( "M_CosThetaH", "M vs. cos#vartheta_{H}", 180, lowMass, highMass, 200, -1, 1);
	TH2F* M_PhiH = new TH2F( "M_PhiH", "M vs. #varphi_{H}", 180, lowMass, highMass, 200, -3.14, 3.14);
	TH2F* M_Phi_Prod = new TH2F( "M_Phi_Prod", "M vs. #Phi_{Prod}", 180, lowMass, highMass, 200, -3.14, 3.14);

	int eventCounter = 0;
	while( eventCounter < nEvents ){

		if( batchSize < 1E4 ){
			cout << "WARNING:  small batches could have batch-to-batch variations\n"
			     << "          due to different maximum intensities!" << endl;
		}

		cout << "Generating four-vectors..." << endl;

		// decay vector (and lowerVertex, if generated)
		ati.clearEvents();
		int i=0;
		while( i < batchSize ){

			double weight = 1.;

			double vec_mass_bw = m_bwGen[0]().first;
			if( fabs(vec_mass_bw - vecMass) > 2.5*vecWidth )
				continue;
			// make sure generated BW is not below threshold of vector->2PS
			double vecthreshold=0;
			for(unsigned int m=0; m<vectorMasses.size(); m++){
				vecthreshold+=vectorMasses[m];
			}
			if(vec_mass_bw<vecthreshold)
				continue;

			// set new production threshold according to generated vector mass
			threshold = childMasses[0];
		    threshold += vec_mass_bw;
			resProd.getProductionMechanism().setMassRange( threshold<lowMass ? lowMass : threshold,highMass );

			//Avoids Tcm < 0 in NBPhaseSpaceFactory and BWgenerator

			vector<double> childMasses_vec_bw;
            childMasses_vec_bw.push_back(childMasses[0]);
        	childMasses_vec_bw.push_back(vec_mass_bw);

			resProd.setChildMasses(childMasses_vec_bw);

			// setup lower vertex decay
			pair< double, double > bwLowerVertex;
			double lowerVertex_mass_bw = 0.;
			if(bwGenLowerVertex.size() == 1) {
				bwLowerVertex = bwGenLowerVertex[0]();
				lowerVertex_mass_bw = bwLowerVertex.first;
				weight *= bwLowerVertex.second;
				if ( lowerVertex_mass_bw < thresholdLowerVertex || lowerVertex_mass_bw > 2.0) continue;
				resProd.getProductionMechanism().setRecoilMass( lowerVertex_mass_bw );
			}

			  Kinematics* step1 = resProd.generate();
			  TLorentzVector beam = step1->particle( 0 );
			  TLorentzVector recoil = step1->particle( 1 );
			  TLorentzVector bachelor = step1->particle( 2 );
			  TLorentzVector vec = step1->particle( 3 );
			  TLorentzVector vec_ps = bachelor + vec;

			  // decay step for vector
        	  NBodyPhaseSpaceFactory vec_to_ps = NBodyPhaseSpaceFactory( vec_mass_bw, vectorMasses);
			  vector<TLorentzVector> vec_daughters = vec_to_ps.generateDecay();
			  vector<TLorentzVector> vec_boosted_daughters;

			  for(uint idaught = 0; idaught<vec_daughters.size(); idaught++) {
				  TLorentzVector vec_boosted_daughter = vec_daughters[idaught];
				  vec_boosted_daughter.Boost( vec.BoostVector() );
				  vec_boosted_daughters.push_back(vec_boosted_daughter);
			  }

			  // decay step for lowerVertex
			  TLorentzVector nucleon;
			  vector<TLorentzVector> lowerVertexChild;
			  if(bwGenLowerVertex.size() == 1) {
				  NBodyPhaseSpaceFactory lowerVertex_decay = NBodyPhaseSpaceFactory( lowerVertex_mass_bw, massesLowerVertex);
				  lowerVertexChild = lowerVertex_decay.generateDecay();

				  // boost to lab frame via recoil kinematics
				  for(unsigned int j=0; j<lowerVertexChild.size(); j++)
					  lowerVertexChild[j].Boost( recoil.BoostVector() );
				  nucleon = lowerVertexChild[0];
			  }
			  else
				  nucleon = recoil;


			  // store particles in kinematic class
			  vector< TLorentzVector > allPart;
			  //same order as config file, Vec_ps_refl amplitudes and AmpTools kin Tree
			  allPart.push_back( beam );
			  allPart.push_back( nucleon );
			  allPart.push_back( bachelor );
			  for(uint idaught = 0; idaught<vec_boosted_daughters.size(); idaught++)
				  allPart.push_back( vec_boosted_daughters[idaught] );
			  if(bwGenLowerVertex.size() == 1)
				  for(unsigned int j=1; j<lowerVertexChild.size(); j++)
                                        allPart.push_back(lowerVertexChild[j]);

			  weight *= step1->weight();
			  Kinematics* kin = new Kinematics( allPart, weight );
			  ati.loadEvent( kin, i, batchSize );
			  delete step1;
			  delete kin;
			  i++;
    		}

		cout << "Processing events..." << endl;

		// include factor of 1.5 to be safe in case we miss peak -- avoid
		// intensity calculation of we are generating flat data
		double maxInten = ( genFlat ? 1 : 1.50* ati.processEvents( reaction->reactionName() ) );


		for( int i = 0; i < batchSize; ++i ){

			Kinematics* evt = ati.kinematics( i );
			TLorentzVector resonance;
			for (unsigned int j=2; j<Particles.size(); j++)
			  resonance += evt->particle( j );

			TLorentzVector isobar;
			for (unsigned int j=3; j<Particles.size(); j++)
			  isobar += evt->particle( j );

			TLorentzVector isobar2;
			for (unsigned int j=4; j<Particles.size(); j++)
			  isobar2 += evt->particle( j );

			TLorentzVector recoil = evt->particle( 1 );
            if(bwGenLowerVertex.size()) {
				for(unsigned int j=Particles.size(); j<evt->particleList().size(); j++)
                                        recoil += evt->particle( j );
            }

			double genWeight = evt->weight();

			// cannot ask for the intensity if we haven't called process events above
			double weightedInten = ( genFlat ? 1 : ati.intensity( i ) );
			// cout << " i=" << i << "  intensity_i=" << weightedInten << endl;

			if( !diag ){

				// obtain this by looking at the maximum value of intensity * genWeight
				double rand = gRandom->Uniform() * maxInten;

				if( weightedInten > rand || genFlat ){

					mass->Fill( resonance.M() );
					massW->Fill( resonance.M(), genWeight );

					intenW->Fill( weightedInten );
					intenWVsM->Fill( resonance.M(), weightedInten );

					M_isobar->Fill( isobar.M() );
					M_isobar2->Fill( isobar2.M() );
					M_recoil->Fill( recoil.M() );
					M_recoilW->Fill( recoil.M(), weightedInten );

					// calculate angular variables
                    Int_t numparticles = evt->particleList().size();
					TLorentzVector beam = evt->particle ( 0 );
					TLorentzVector p1 = evt->particle ( 2 );
					TLorentzVector p2 = evt->particle ( 3 );
					TLorentzVector p3 = evt->particle ( 4 );
                    TLorentzVector p4;
                    if(numparticles==6)
                        p4 = evt->particle ( 5 );
					TLorentzVector target(0,0,0,ParticleMass(Proton));

					M_p1->Fill( p1.M() );
					M_p2->Fill( p2.M() );
					M_p3->Fill( p3.M() );
					M_p4->Fill( p4.M() );

					double dalitz_s, dalitz_t, dalitz_u, dalitz_d, dalitz_sc, dalitzx, dalitzy;
					dalitz_s = (p3+p4).M2();//s=M(pip pim)
					dalitz_t = (p2+p3).M2();//s=M(pip pi0)
					dalitz_u = (p2+p4).M2();//s=M(pim pi0)
					dalitz_d = 2*(p2+p3+p4).M()*( (p2+p3+p4).M() - ((2*0.13957018)+0.1349766) );
					dalitz_sc = (1/3.)*( (p2+p3+p4).M2() + ((2*(0.13957018*0.13957018))+(0.1349766*0.1349766)) );
					dalitzx = sqrt(3.)*(dalitz_t - dalitz_u)/dalitz_d;
					dalitzy = 3.*(dalitz_sc - dalitz_s)/dalitz_d;
					M_dalitz->Fill(dalitzx,dalitzy);

					t->Fill(-1*(recoil-target).M2());

                    TLorentzVector Gammap = beam + target;
                    vector <double> loccosthetaphi = getomegapiAngles(polAngle, isobar, resonance, beam, Gammap);
                    double cosTheta = cos(loccosthetaphi[0]);
                    double phi = loccosthetaphi[1];

                    vector <double> loccosthetaphih = getomegapiAngles( p3, isobar, resonance, Gammap, p4);
                    double cosThetaH = cos(loccosthetaphih[0]);
                    double phiH = loccosthetaphih[1];

					M_CosTheta->Fill( resonance.M(), cosTheta);
					M_Phi->Fill( resonance.M(), phi);
					M_CosThetaH->Fill( resonance.M(), cosThetaH);
					M_PhiH->Fill( resonance.M(), phiH);

					double lambda_omega = loccosthetaphih[2];
					lambda->Fill(lambda_omega);

                    double Phi = loccosthetaphi[2];
					M_Phi_Prod->Fill( resonance.M(), Phi);

                    GDouble psi = phi - Phi;
                    if(psi < -1*PI) psi += 2*PI;
                    if(psi > PI) psi -= 2*PI;

					CosTheta_psi->Fill( psi, cosTheta);

					// we want to save events with weight 1
					evt->setWeight( 1.0 );

                    #ifdef HDDM
					if( hddmOut ) hddmOut->writeEvent( *evt, pTypes );
                    #endif
					rootOut->writeEvent( *evt );
					++eventCounter;
					if(eventCounter >= nEvents) break;
				}
			}
			else{

				mass->Fill( resonance.M() );
				massW->Fill( resonance.M(), genWeight );

				intenW->Fill( weightedInten );
				intenWVsM->Fill( resonance.M(), weightedInten );

				++eventCounter;
			}

			delete evt;
		}

		cout << eventCounter << " events were processed." << endl;
	}

	mass->Write();
	massW->Write();
	intenW->Write();
	intenWVsM->Write();
	M_isobar->Write();
	M_isobar2->Write();
	M_recoil->Write();
	M_recoilW->Write();
    M_p1->Write();
    M_p2->Write();
    M_p3->Write();
    M_p4->Write();
    M_dalitz->Write();
	lambda->Write();
	t->Write();
	CosTheta_psi->Write();
	M_CosTheta->Write();
	M_Phi->Write();
	M_CosThetaH->Write();
	M_PhiH->Write();
	M_Phi_Prod->Write();

	diagOut->Close();

    #ifdef HDDM
	if( hddmOut ) delete hddmOut;
    #endif
	delete rootOut;

}
