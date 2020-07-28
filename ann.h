#ifndef ann_h
#define ann_h

#include <stdint.h>
#include <cmath>

#include "machinelearning.h"

namespace machinelearning {
namespace ann {

class ann : public MachineLearning {
 public:
  ann(char* configuration_file, double learn_rate_ = 0.01,
      double momentum_ = 0.2, double max_epoch_ = 5000,
      double max_weight_for_init_ = 0.3, double target_error_ = 0.000000000001,
      int num_layer_ = 3);
  void Train(char*);
  std::vector<int> Predict(char*, bool);

 protected:
  std::vector<std::vector<double> > input_train_instances_;
  // store the whole input of training instances
  std::vector<std::vector<double> > output_train_instances_;  
  // store the whole output of training instances

  // Neural Network structure
  int num_neuron_layer1_;
  int num_neuron_layer2_;
  int num_neuron_layer3_;
  int num_layer_;

  // Neural Network parameters
  std::vector<double> layer2_parameters_;
  std::vector<double> layer3_parameters_;
  std::vector<std::vector<std::vector<double> > > weight_of_network_;
  std::vector<std::vector<std::vector<double> > > weight_delta_of_network_;
  std::vector<std::vector<double> > delta_gradient_of_network_;

  // Neural Netowrk attributes
  double target_error_;
  double learn_rate_;
  double momentum_;
  double max_weight_for_init_;
  int64_t max_epoch_;

  void ParseConfiguration(char*);        // read configuration file
  void PrepareMemoryForTrainingData();   // allocate memory for training
                                         // data
  void StoreTrainingData(char*);         // store training data
  void InitNetworkParameter();           // initialize the netowrk
  void PrintNetworkParameter();     // print out network parameter for debug
  void OptimizeNetworkParameter();  // optimize the network parameter via
                                    // training data
  int DoOnePrediction(std::vector<double>&);
  // calculate the probability of each choice and choose the greatest one as our
  // prediction

  double inline Sigmoid(double x) {
    x = exp(-x);
    x = 1 / (1 + x);
    return x;
  }
};

}  // end of namespace ann
}  // namespace machinelearning

#endif
