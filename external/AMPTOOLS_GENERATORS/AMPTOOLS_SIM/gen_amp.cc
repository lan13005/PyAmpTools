
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

#include "AMPTOOLS_AMPS/TwoPiAngles.h"
#include "AMPTOOLS_AMPS/BreitWigner.h"
#include "AMPTOOLS_AMPS/Piecewise.h"
#include "AMPTOOLS_AMPS/PhaseOffset.h"
#include "AMPTOOLS_AMPS/Zlm.h"

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

#include "AMPTOOLS_SIM/gen_amp.h"

using std::complex;
using namespace std;

gen_amp::gen_amp()
{
}

gen_amp::~gen_amp()
{
}

void
gen_amp::print_help(){
	cout << endl << "GEN_AMP: Simulation attributes:" << endl << endl;
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
gen_amp::generate(){

	if( configfile.size() == 0 || outname.size() == 0 ){
		cout << "No config file or output specificed:  run gen_amp -h for help" << endl;
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
	vector<double> childMasses;
	double threshold = 0;
	unsigned int maxUpperVertexChild = reaction->particleList().size();
	// don't include non-nucleon lower vertex decay particles in meson decay
	if(bwGenLowerVertex.size() == 1) maxUpperVertexChild -= (ParticlesLowerVertex.size()-1);
	for (unsigned int i = 0; i < maxUpperVertexChild; i++){
	  Particle_t locEnum = ParticleEnum(reaction->particleList()[i].c_str());
	  // Beam particle is always photon
	  if (locEnum == 0 && i > 0)
	    cout << "ConfigFileParser WARNING:  unknown particle type \"" << reaction->particleList()[i] << "\"" << endl;
	  Particles.push_back(ParticleEnum(reaction->particleList()[i].c_str()));
	  if (i>1){
	    childMasses.push_back(ParticleMass(Particles[i]));
	    threshold += ParticleMass(Particles[i]);
	  }
	}

	//switch recoil particle
	ProductionMechanism::Recoil recoil;
	bool isBaryonResonance = false;
	switch(Particles[1]){
	case Proton:
	  recoil = ProductionMechanism::kProton;
	  break;
	case Neutron:
	  recoil = ProductionMechanism::kNeutron;
	  break;
	case Pb208:
	  recoil = ProductionMechanism::kZ;
	  break;
	case PiPlus:
	case PiMinus: // works like an OR statement
	  recoil = ProductionMechanism::kPion;
	  isBaryonResonance = true;
	  break;
	case KPlus:
	case KMinus:
	  recoil = ProductionMechanism::kKaon;
	  isBaryonResonance = true;
	  break;
	default:
	  cout << "ConfigFileParser WARNING: not supported recoil particle type \"" << reaction->particleList()[1].c_str()
	       << "\", defaulted to Proton" << endl;
	  recoil = ProductionMechanism::kProton;
	}

	// loop to look for resonance in config file
	// currently only one at a time is supported
	const vector<ConfigFileLine> configFileLines = parser.getConfigFileLines();
	double resonance[]={1.0, 1.0};
	bool foundResonance = false;
	for (vector<ConfigFileLine>::const_iterator it=configFileLines.begin(); it!=configFileLines.end(); it++) {
	  if ((*it).keyword() == "define") {
		if ((*it).arguments()[0] == "rho" || (*it).arguments()[0] == "omega" || (*it).arguments()[0] == "phi" || (*it).arguments()[0] == "b1" || (*it).arguments()[0] == "a1" || (*it).arguments()[0] == "Lambda1520"){
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
		cout << "ConfigFileParser WARNING:  no known resonance found, seed with mass = width = 1GeV" << endl;

	// random number initialization (set to 0 by default)
	TRandom3* gRandom = new TRandom3();
	gRandom->SetSeed(seed);
	seed = gRandom->GetSeed();
	cout << "TRandom3 Seed : " << seed << endl;

	// setup AmpToolsInterface
	AmpToolsInterface::registerAmplitude( TwoPiAngles() );
	AmpToolsInterface::registerAmplitude( BreitWigner() );
	AmpToolsInterface::registerAmplitude( Piecewise() );
	AmpToolsInterface::registerAmplitude( PhaseOffset() );
	AmpToolsInterface::registerAmplitude( Zlm() );
	AmpToolsInterface ati( cfgInfo, AmpToolsInterface::kMCGeneration );

	// loop to look for beam configuration file
	TString beamConfigFile;
	const vector<ConfigFileLine> configFileLinesBeam = parser.getConfigFileLines();
	for (vector<ConfigFileLine>::const_iterator it=configFileLinesBeam.begin(); it!=configFileLinesBeam.end(); it++) {
		if ((*it).keyword() == "define") {
			TString beamArgument =  (*it).arguments()[0].c_str();
			if(beamArgument.Contains("beamconfig")) {
				beamConfigFile = (*it).arguments()[1].c_str();
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
		( genFlat ? ProductionMechanism::kFlat : ProductionMechanism::kResonant );

	// generate over a range of mass
	// start with threshold or lowMass, whichever is higher
	GammaPToNPartP resProd;
	double minMass = (threshold < lowMass ? lowMass : threshold);
	resProd = GammaPToNPartP( minMass, highMass, childMasses, recoil, type, slope, lowT, highT, seed, beamConfigFile );

	if (childMasses.size() < 2){
		cout << "ConfigFileParser ERROR:  single particle production is not yet implemented" << endl;
		return;
	}

	double targetMass = ParticleMass(ParticleEnum("Proton"));
	if(recoil == ProductionMechanism::kZ)
		targetMass = ParticleMass(Particles[1]);
	double recMass = ParticleMass(Particles[1]);
	double cmEnergy = sqrt(targetMass*(targetMass + 2*beamLowE));
	if ( cmEnergy < minMass + recMass ){
		cout << "ConfigFileParser ERROR:  Minimum photon energy not high enough to create resonance!" << endl;
		return;
	}
	else if ( cmEnergy < highMass + recMass )
		cout << "ConfigFileParser WARNING:  Minimum photon energy not high enough to guarantee flat mass distribution!" << endl;

	// seed the distribution with a sum of noninterfering Breit-Wigners
	// we can easily compute the PDF for this and divide by that when
	// doing accept/reject -- improves efficiency if seeds are picked well

	if( !genFlat ){

		// the lines below should be tailored by the user for the particular desired
		// set of amplitudes -- doing so will improve efficiency.  Leaving as is
		// won't make MC incorrect, it just won't be as fast as it could be

		resProd.addResonance( resonance[0], resonance[1],  1.0 );
	}

	vector< int > pTypes;
	for (unsigned int i=0; i<Particles.size(); i++)
		pTypes.push_back( Particles[i] );
	for (unsigned int i=0; i<ParticlesLowerVertex.size(); i++) {
		if(ParticlesLowerVertex[i] == Proton || ParticlesLowerVertex[i] == Neutron) continue;
        	pTypes.push_back( ParticlesLowerVertex[i] );
	}

	#ifdef HDDM
	HDDMDataWriter* hddmOut = NULL;
	if( hddmname.size() != 0 ) hddmOut = new HDDMDataWriter( hddmname, runNum, seed);
	#endif

	// the first argument to the FSRootDataWriter is the number of particles *in addition to* the beam
	// particle, which is typically the first in the list in GlueX reaction definitions
	DataWriter* rootOut = ( fsRootFormat ?
				static_cast< DataWriter*>( new FSRootDataWriter( reaction->particleList().size()-1, outname ) ) :
				static_cast< DataWriter* >( new ROOTDataWriter( outname ) ) );

	TFile* diagOut = new TFile( "gen_amp_diagnostic.root", "recreate" );
	ostringstream locStream;
	ostringstream locIsobarStream;
	for (unsigned int i=2; i<Particles.size(); i++){
	  locStream << ParticleName_ROOT(Particles[i]);
	  if ( i> 2 )
	    locIsobarStream << ParticleName_ROOT(Particles[i]);
	}
	string locHistTitle = string("Resonance Mass ;") + locStream.str() + string(" Invariant Mass (GeV/c^{2});");
	string locIsobarTitle = string("Isobar Mass ;") + locIsobarStream.str() + string(" Invariant Mass (GeV/c^{2});");

	TH1F* mass = new TH1F( "M", locHistTitle.c_str(), 180, lowMass, highMass );
	TH1F* massW = new TH1F( "M_W", ("Weighted "+locHistTitle).c_str(), 180, lowMass, highMass );
	massW->Sumw2();
	TH1F* intenW = new TH1F( "intenW", "True PDF / Gen. PDF", 1000, 0, 100 );
	TH2F* intenWVsM = new TH2F( "intenWVsM", "Ratio vs. M", 100, lowMass, highMass, 1000, 0, 10 );

	TH1F* t = new TH1F( "t", "-t Distribution", 200, 0, 2 );

	TH1F* E = new TH1F( "E", "Beam Energy", 120, 0, 12 );
	TH2F* EvsM = new TH2F( "EvsM", "Beam Energy vs Mass", 120, 0, 12, 180, lowMass, highMass );

	TH1F* M_isobar = new TH1F( "M_isobar", locIsobarTitle.c_str(), 200, 0, 2 );
	TH1F* M_recoil = new TH1F( "M_recoil", "; Recoil mass (GeV)", 200, 0, 2 );

	TH2F* CosTheta_psi = new TH2F( "CosTheta_psi", "cos#theta vs. #psi", 180, -3.14, 3.14, 100, -1, 1);
	TH2F* M_CosTheta = new TH2F( "M_CosTheta", "M vs. cos#vartheta", 180, lowMass, highMass, 200, -1, 1);
	TH2F* M_Phi = new TH2F( "M_Phi", "M vs. #varphi", 180, lowMass, highMass, 200, -3.14, 3.14);
	TH2F* M_Phi_lab = new TH2F( "M_Phi_lab", "M vs. #varphi", 180, lowMass, highMass, 200, -3.14, 3.14);

	int eventCounter = 0;
	while( eventCounter < nEvents ){

		if( batchSize < 1E4 ){
			cout << "WARNING:  small batches could have batch-to-batch variations\n"
			     << "          due to different maximum intensities!" << endl;
		}

		cout << "Generating four-vectors..." << endl;

		ati.clearEvents();
		int i=0;
        while( i < batchSize ){

			double weight = 1.;

			Kinematics* kin;
			if(bwGenLowerVertex.size() == 0)
				kin = resProd.generate(); // stable particle at lower vertex
			else {
				// unstable particle at lower vertex
				pair< double, double > bwLowerVertex = bwGenLowerVertex[0]();
				double lowerVertex_mass_bw = bwLowerVertex.first;
				weight *= bwLowerVertex.second;

				if ( lowerVertex_mass_bw < thresholdLowerVertex || lowerVertex_mass_bw > 2.0) continue;
				resProd.getProductionMechanism().setRecoilMass( lowerVertex_mass_bw );

				Kinematics* step1 = resProd.generate();
				TLorentzVector beam = step1->particle( 0 );
				TLorentzVector recoil = step1->particle( 1 );

				// loop over meson decay
				vector<TLorentzVector> mesonChild;
				for(unsigned int i=0; i<childMasses.size(); i++)
					mesonChild.push_back(step1->particle( 2+i ));

				// decay step for lower vertex
				TLorentzVector nucleon; // proton or neutron
				NBodyPhaseSpaceFactory lowerVertex_decay = NBodyPhaseSpaceFactory( lowerVertex_mass_bw, massesLowerVertex);
				vector<TLorentzVector> lowerVertexChild = lowerVertex_decay.generateDecay();
				// boost to lab frame via recoil kinematics
				for(unsigned int j=0; j<lowerVertexChild.size(); j++)
				  lowerVertexChild[j].Boost( recoil.BoostVector() );
				nucleon = lowerVertexChild[0];

				// store particles in kinematic class
				vector< TLorentzVector > allPart;
				allPart.push_back( beam );
				allPart.push_back( nucleon );
				// loop over meson decay particles
				for(unsigned int j=0; j<mesonChild.size(); j++)
					allPart.push_back(mesonChild[j]);
				// loop over lower vertex decay particles
				for(unsigned int j=1; j<lowerVertexChild.size(); j++)
					allPart.push_back(lowerVertexChild[j]);

				weight *= step1->weight();
				kin = new Kinematics( allPart, weight );
				delete step1;
			}

			ati.loadEvent( kin, i, batchSize );
			delete kin;
			i++;
		}

		cout << "Processing events..." << endl;

		// include factor of 1.5 to be safe in case we miss peak -- avoid
		// intensity calculation of we are generating flat data
		double maxInten = ( genFlat ? 1 : 1.5 * ati.processEvents( reaction->reactionName() ) );


		for( int i = 0; i < batchSize; ++i ){

			Kinematics* evt = ati.kinematics( i );
			TLorentzVector resonance;
			for (unsigned int j=2; j<Particles.size(); j++)
			  resonance += evt->particle( j );

			TLorentzVector isobar;
			for (unsigned int j=3; j<Particles.size(); j++)
			  isobar += evt->particle( j );

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
					M_recoil->Fill( recoil.M() );

					// calculate angular variables
					TLorentzVector beam = evt->particle ( 0 );
					TLorentzVector rec = evt->particle ( 1 );
					TLorentzVector p1 = evt->particle ( 2 );
					TLorentzVector target(0,0,0,rec[3]);

					if(isBaryonResonance) // assume t-channel
						t->Fill(-1*(beam-evt->particle(1)).M2());
					else
						t->Fill(-1*(recoil-target).M2());

					E->Fill(beam.E());
					EvsM->Fill(beam.E(),resonance.M());

					TLorentzRotation resonanceBoost( -resonance.BoostVector() );

					TLorentzVector beam_res = resonanceBoost * beam;
					TLorentzVector rec_res = resonanceBoost * rec;
					TLorentzVector p1_res = resonanceBoost * p1;

					// normal to the production plane
                    TVector3 y = (beam.Vect().Unit().Cross(-rec.Vect().Unit())).Unit();

                    // choose helicity frame: z-axis opposite recoil proton in rho rest frame
                    TVector3 z = -1. * rec_res.Vect().Unit();
                    TVector3 x = y.Cross(z).Unit();
                    TVector3 angles( (p1_res.Vect()).Dot(x),
                                     (p1_res.Vect()).Dot(y),
                                     (p1_res.Vect()).Dot(z) );

                    double cosTheta = angles.CosTheta();
                    double phi = angles.Phi();

					M_CosTheta->Fill( resonance.M(), cosTheta);
					M_Phi->Fill( resonance.M(), phi);
					M_Phi_lab->Fill( resonance.M(), rec.Phi());

					TVector3 eps(1.0, 0.0, 0.0); // beam polarization vector
                                        double Phi = atan2(y.Dot(eps), beam.Vect().Unit().Dot(eps.Cross(y)));

                                        GDouble psi = phi - Phi;
                                        if(psi < -1*PI) psi += 2*PI;
                                        if(psi > PI) psi -= 2*PI;

					CosTheta_psi->Fill( psi, cosTheta);

					// we want to save events with weight 1
					evt->setWeight( 1.0 );

					#ifdef HDDM
					if( hddmOut ) hddmOut->writeEvent( *evt, pTypes, centeredVertex );
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
				TLorentzVector rec = evt->particle ( 1 );

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
	M_recoil->Write();
	t->Write();
	E->Write();
	EvsM->Write();
	CosTheta_psi->Write();
	M_CosTheta->Write();
	M_Phi->Write();
	M_Phi_lab->Write();

	diagOut->Close();

	#ifdef HDDM
	if( hddmOut ) delete hddmOut;
	#endif
	delete rootOut;

	delete mass;
	delete massW;
	delete intenW;
	delete intenWVsM;
	delete M_isobar;
	delete M_recoil;
	delete t;
	delete E;
	delete EvsM;
	delete CosTheta_psi;
	delete M_CosTheta;
	delete M_Phi;
	delete M_Phi_lab;
	delete diagOut;

	delete gRandom;
	delete cfgInfo;
}
