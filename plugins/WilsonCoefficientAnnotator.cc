// -*- C++ -*-
//
// Package:    six-dimensional-ttV/SixDimensionalTTV
// Class:      WilsonCoefficientAnnotator
//
/**\class WilsonCoefficientAnnotator WilsonCoefficientAnnotator.cc six-dimensional-ttV/SixDimensionalTTV/plugins/WilsonCoefficientAnnotator.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Anna Elizabeth Woodard
//         Created:  Sun, 03 Jul 2016 18:05:24 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class declaration
//

class WilsonCoefficientAnnotator : public edm::one::EDProducer<edm::BeginRunProducer> {
   public:
      explicit WilsonCoefficientAnnotator(const edm::ParameterSet&);
      ~WilsonCoefficientAnnotator();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void beginRunProduce(edm::Run& run, edm::EventSetup const& es) override;
      virtual void endJob() override;

      // ----------member data ---------------------------

      std::vector<double> point_;
      std::vector<std::string> wilsonCoefficients_;
      std::string process_;
};


WilsonCoefficientAnnotator::WilsonCoefficientAnnotator(const edm::ParameterSet& iConfig):
  point_(iConfig.getParameter<std::vector<double>>("point")),
  wilsonCoefficients_(iConfig.getParameter<std::vector<std::string>>("wilsonCoefficients")),
  process_(iConfig.getParameter<std::string>("process"))
{

  produces<std::vector<double>, edm::InRun>("point");
  produces<std::vector<std::string>, edm::InRun>("wilsonCoefficients");
  produces<std::string, edm::InRun>("process");

}


WilsonCoefficientAnnotator::~WilsonCoefficientAnnotator()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
WilsonCoefficientAnnotator::beginRunProduce(edm::Run& run, edm::EventSetup const& es)
{
   using namespace edm;
   std::auto_ptr<std::vector<double>> point(new std::vector<double>);
   *point = point_;
   std::auto_ptr<std::vector<std::string>> wilsonCoefficients(new std::vector<std::string>);
   *wilsonCoefficients = wilsonCoefficients_;
   std::auto_ptr<std::string> process(new std::string);
   *process = process_;

   run.put(point, "point");
   run.put(wilsonCoefficients, "wilsonCoefficients");
   run.put(process, "process");
}

void
WilsonCoefficientAnnotator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
}

// ------------ method called once each job just before starting event loop  ------------
void
WilsonCoefficientAnnotator::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
WilsonCoefficientAnnotator::endJob() {
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
WilsonCoefficientAnnotator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(WilsonCoefficientAnnotator);
