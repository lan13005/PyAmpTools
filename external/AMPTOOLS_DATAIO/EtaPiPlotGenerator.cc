#include "TLorentzVector.h"
#include "TLorentzRotation.h"
#include "TLorentzVector.h"
#include "TLorentzRotation.h"

#include "EtaPiPlotGenerator.h"
#include "IUAmpTools/Histogram1D.h"
#include "IUAmpTools/Kinematics.h"

EtaPiPlotGenerator::EtaPiPlotGenerator( const FitResults& results ) :
PlotGenerator( results )
{
	createHistograms();
}

EtaPiPlotGenerator::EtaPiPlotGenerator( ) :
PlotGenerator( )
{
	createHistograms();
}

void EtaPiPlotGenerator::createHistograms() {
  // calls to bookHistogram go here

  bookHistogram( kEtaPiMass, new Histogram1D( 125, 0.8, 1.8, "Metapi", "Invariant Mass of #eta #pi") );
  bookHistogram( kEtaPiMass_40MeVBin, new Histogram1D( 25, 0.8, 1.8, "Metapi_40MeVBin", "Invariant Mass of #eta #pi") );
  //bookHistogram( kEtaPiMass, new Histogram1D( 100, 1.04, 1.8, "Metapi", "Invariant Mass of #eta #pi") );
  //bookHistogram( kEtaPiMass_40MeVBin, new Histogram1D( 19, 1.04, 1.8, "Metapi_40MeVBin", "Invariant Mass of #eta #pi") );
  bookHistogram( kEtaProtonMass, new Histogram1D( 90, 1.4, 4.0, "Metap", "Invariant Mass of #eta p") );
  bookHistogram( kPi0ProtonMass,  new Histogram1D( 90, 1.0, 3.5, "Mpip", "Invariant Mass of #pi #p") );

  bookHistogram( kEtaCosTheta, new Histogram1D( 50, -1., 1., "cosTheta", "cos( #theta ) of Resonance Production") );
  bookHistogram( kPhiPi,  new Histogram1D( 50, -1*PI, PI, "PhiPi",  "#Phi_{#pi}" ) );
  bookHistogram( kPhiEta, new Histogram1D( 50, -1*PI, PI, "PhiPiEta", "#Phi_{#eta}" ) );
  bookHistogram( kPhi, new Histogram1D( 50, -1*PI, PI, "Phi", "#Phi" ) );
  bookHistogram( kphi, new Histogram1D( 50, -1*PI, PI, "phi", "#phi" ) );
  bookHistogram( kPsi, new Histogram1D( 50, -1*PI, PI, "psi", "#psi" ) );
  bookHistogram( kt, new Histogram1D( 100, 0, 1.0, "t", "-t" ) );
}

void
EtaPiPlotGenerator::projectEvent( Kinematics* kin, const string& reactionName ){
  //cout << "START PROJECT EVENT PLOT GENERATOR" << endl;
  //cout << cfgInfo()->amplitudeList( reactionName, "", "").at(0)->factors().at(0).at(1) << endl;
  //cout << cfgInfo()->amplitudeList( reactionName, "", "").at(0)->factors().at(0).at(2) << endl;
  //cout << cfgInfo()->amplitudeList( reactionName, "", "").at(0)->factors().at(0).at(3) << endl;
  //cout << cfgInfo()->amplitudeList( reactionName, "", "").at(0)->factors().at(0).at(4) << endl;
  //cout << cfgInfo()->amplitudeList( reactionName, "", "").at(0)->factors().at(0).at(5) << endl;
  //cout << cfgInfo()->amplitudeList( reactionName, "", "").at(0)->factors().at(0).at(5) << endl;
  double polAngle=stod(cfgInfo()->amplitudeList( reactionName, "", "").at(0)->factors().at(0).at(5));
  //cout << polAngle << endl;

  // Our reaction has the order 14 7 17 when using tree_to_amptools. so its proton pi0 eta
  TLorentzVector beam   = kin->particle( 0 );
  TLorentzVector recoil = kin->particle( 1 );
  TLorentzVector p1 = kin->particle( 2 );
  TLorentzVector p2 = kin->particle( 3 );
  TLorentzVector target(0,0,0,0.9382719);
  TLorentzVector resonance = p1 + p2;

  /// ************************************************
  // This should be calculating the angles in the helictiy frame
  /// ************************************************
  //TLorentzRotation resonanceBoost( -resonance.BoostVector() );
  //TLorentzVector recoil_res = resonanceBoost * recoil;
  //TLorentzVector p1_res = resonanceBoost * p1;
  //TLorentzVector p2_res = resonanceBoost * p2;
  //// normal to the production plane
  //TVector3 y = (beam.Vect().Unit().Cross(-recoil.Vect().Unit())).Unit();
  ////// z-axis opposite recoil proton in rho rest frame
  //TVector3 z = -1. * recoil_res.Vect().Unit();
  //TVector3 x = y.Cross(z).Unit();

  /// ************************************************
  // This should be calculating the angles in the GJ frame
  /// ************************************************
  TLorentzVector cm=beam+target;
  TLorentzRotation cmBoost( -cm.BoostVector() );
  TLorentzVector beam_cm=cmBoost*beam;
  TLorentzVector recoil_cm=cmBoost*recoil;
  TLorentzVector pi0_cm=cmBoost*p1;
  TLorentzVector eta_cm=cmBoost*p2;
  TLorentzVector resonance_cm=pi0_cm+eta_cm;

  TLorentzRotation resonanceBoost( -resonance_cm.BoostVector() );
  TLorentzVector beam_res = resonanceBoost*beam_cm;
  TLorentzVector recoil_res = resonanceBoost*recoil_cm;
  TLorentzVector eta_res = resonanceBoost*eta_cm;
  TLorentzVector pi0_res = resonanceBoost*pi0_cm;


  /////////////////////
  // MY OLD CODE
  //TVector3 z = beam_res.Vect().Unit();
  //// this y should be the normal of the production plane. If we do a boost in a direction in the production plane the perp direction doesn't change. We could use the beam and the recoiled proton to define the
  //// production plane in this new frame. Let us define it in the CM frame.
  //TVector3 y = resonance_cm.Vect().Cross(beam_cm.Vect()).Unit();
  //////////////////////
  /////////////////////
  // NEW CODE TAKEN FROM ZLM.CC
  TVector3 z = -1. * recoil_res.Vect().Unit();
  // or GJ frame?
  // TVector3 z = beam_res.Vect().Unit();
  // normal to the production plane
  TVector3 y = (beam.Vect().Unit().Cross(-recoil.Vect().Unit())).Unit();
  /////////////////////
  TVector3 x = y.Cross(z).Unit();
  TVector3 angles(   (eta_res.Vect()).Dot(x),
                     (eta_res.Vect()).Dot(y),
                     (eta_res.Vect()).Dot(z) );
  GDouble cosTheta = angles.CosTheta();

  GDouble phi = angles.Phi();

  TVector3 eps(TMath::Cos(polAngle*0.01745), TMath::Sin(polAngle*0.01745), 0.0); // beam polarization vector
  GDouble Phi = atan2(y.Dot(eps), beam.Vect().Unit().Dot(eps.Cross(y)));

  GDouble psi = phi - Phi;
  if(psi < -1*PI) psi += 2*PI;
  if(psi > PI) psi -= 2*PI;

  // compute invariant t
  GDouble t = - 2* recoil.M() * (recoil.E()-recoil.M());

  // calls to fillHistogram go here
  //cout << "BEGIN FILL" << endl;
  fillHistogram( kEtaPiMass, ( resonance ).M() );
  fillHistogram( kEtaPiMass_40MeVBin, ( resonance ).M() );
  fillHistogram( kEtaProtonMass, (recoil+p2).M() );
  fillHistogram( kPi0ProtonMass, (recoil+p1).M() );
  fillHistogram( kEtaCosTheta, cosTheta );
  fillHistogram( kPhiPi,  p1.Phi() );
  fillHistogram( kPhiEta, p2.Phi() );
  fillHistogram( kPhi, Phi );
  fillHistogram( kphi, phi );
  fillHistogram( kPsi, psi );
  fillHistogram( kt, -t );      // fill with -t to make positive
}
