#include "ann.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>

namespace machinelearning {
namespace ann {

// initialize all the information we need from training data
ann::ann(char *configuration_file, double ilearn_rate_, double imomentum_,
         double imax_epoch_, double imax_weight_for_init_,
         double itarget_error_, int inum_layer_) {
  // set initial value
  learn_rate_ = ilearn_rate_;
  momentum_ = imomentum_;
  max_epoch_ = imax_epoch_;
  max_weight_for_init_ = imax_weight_for_init_;
  target_error_ = itarget_error_;
  num_layer_ = inum_layer_;
  // read configuration
  ParseConfiguration(configuration_file);
}

void ann::Train(char *train_file) {
  PrepareMemoryForTrainingData();
  StoreTrainingData(train_file);

  layer2_parameters_.resize(num_neuron_layer2_ + 1);

  // init bais
  layer2_parameters_[num_neuron_layer2_] = 1;

  layer3_parameters_.resize(num_neuron_layer3_);

  // initialize neural layer parameters
  InitNetworkParameter();
  #ifdef DEBUG
    PrintNetworkParameter();
  #endif
  OptimizeNetworkParameter();
}

void ann::ParseConfiguration(char *configuration_file) {
  // read configuration
  std::ifstream cfg;
  cfg.open(configuration_file);
  if (!cfg) {
    std::cout << "Can't open configure file!" << std::endl;
    return;
  }

  cfg >> num_train_instances_ >> num_test_instances_ >> num_neuron_layer1_ >>
      num_neuron_layer2_ >> num_neuron_layer3_;
  cfg.close();
  #ifdef DEBUG
    std::cout<<num_train_instances_<<num_test_instances_<<num_neuron_layer1_<<num_neuron_layer2_<<num_neuron_layer3_;
  #endif
}

void ann::PrepareMemoryForTrainingData() {
  // prepare memory for input_train_instances_ data
  input_train_instances_.resize(num_train_instances_);
  for (int i = 0; i < num_train_instances_; ++i)
    input_train_instances_[i].resize(num_neuron_layer1_ + 1);

  // prepare memory for output_train_instances_ output
  output_train_instances_.resize(num_train_instances_);
  for (int i = 0; i < num_train_instances_; ++i)
    output_train_instances_[i].resize(num_neuron_layer3_);
}

void ann::StoreTrainingData(char *train_file) {
  std::ifstream trainingDataFile;
  std::string Buf;
  trainingDataFile.open(train_file);
  if (!trainingDataFile) {
    std::cout << "Can't open training data file!" << std::endl;
    return;
  }

  // store input_train_instances_ data and output_train_instances_ output
  for (int i = 0; i < num_train_instances_; ++i) {
    getline(trainingDataFile, Buf);
    std::stringstream lineStream(Buf);
    for (int j = 0; j < num_neuron_layer1_; ++j) {
      getline(lineStream, Buf, ',');
      input_train_instances_[i][j] = stod(Buf);
    }
    for (int j = 0; j < num_neuron_layer3_; ++j) {
      getline(lineStream, Buf, ',');
      output_train_instances_[i][j] = stod(Buf);
    }
    // init bias
    input_train_instances_[i][num_neuron_layer1_] = 1;
  }
}

void ann::InitNetworkParameter() {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(0, 1);

  weight_of_network_.resize(num_layer_ - 1);
  weight_delta_of_network_ .resize(num_layer_ - 1);
  delta_gradient_of_network_.resize(num_layer_ - 1);
  for (int i = 0; i < (num_layer_ - 1); ++i) {
    //TODO merge by changing num_neuron_layer*_ to array
    if (i == 0) {
      weight_of_network_[i].resize(num_neuron_layer1_ + 1);
      weight_delta_of_network_[i].resize(num_neuron_layer1_ + 1);
      delta_gradient_of_network_[i].resize(num_neuron_layer2_, 0);

      for (int j = 0; j <= num_neuron_layer1_; ++j) {
        weight_of_network_[i][j].resize(num_neuron_layer2_ + 1);
        weight_delta_of_network_[i][j].resize(num_neuron_layer2_ + 1, 0);
        for (int k = 0; k < num_neuron_layer2_; ++k)
          weight_of_network_[i][j][k] =
              (2 * (dist(mt) - 0.5) * max_weight_for_init_);
      }
    }
    if (i == 1) {
      weight_of_network_[i].resize(num_neuron_layer2_ + 1);
      weight_delta_of_network_[i].resize(num_neuron_layer2_ + 1);
      delta_gradient_of_network_[i].resize(num_neuron_layer3_, 0);
      for (int j = 0; j <= num_neuron_layer2_; ++j) {
        weight_of_network_[i][j].resize(num_neuron_layer3_);
        weight_delta_of_network_[i][j].resize(num_neuron_layer3_, 0);
        for (int k = 0; k < num_neuron_layer3_; ++k)
          weight_of_network_[i][j][k] =
              (2 * (dist(mt) - 0.5) * max_weight_for_init_);
      }
    }
  }
}

void ann::PrintNetworkParameter() {
  for (int j = 0; j <= num_neuron_layer1_; j++) {
    for (int k = 0; k < num_neuron_layer2_; k++) {
      std::cout << weight_of_network_[0][j][k] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  for (int j = 0; j <= num_neuron_layer2_; j++) {
    for (int k = 0; k < num_neuron_layer3_; k++) {
      std::cout << weight_of_network_[1][j][k] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void ann::OptimizeNetworkParameter() {

#ifdef SHUFFLE_INPUT 
  srand ( unsigned ( std::time(0)) );
  std::vector<int> ilist;
  for(int i=0; i<num_train_instances_ ; ++i) {ilist.push_back(i);}
#endif

  unsigned long long int epoch = 1;
  double error = std::numeric_limits<double>::max();
  double perror = std::numeric_limits<double>::max();
  double deltaerror = std::numeric_limits<double>::max();
  while (epoch <= max_epoch_ && deltaerror > target_error_) {
    // train each pattern for one epoch
    error = 0;
#ifdef SHUFFLE_INPUT 
    random_shuffle(ilist.begin(),ilist.end());
#endif
    for (int i = 0; i < num_train_instances_; ++i) {
      // calculate layer 2 value
      for (int j = 0; j < num_neuron_layer2_; ++j) {
        layer2_parameters_[j] = 0;
        for (int k = 0; k <= num_neuron_layer1_; ++k) {
          layer2_parameters_[j] += (Sigmoid(input_train_instances_[i][k]) *
                                    weight_of_network_[0][k][j]);
        }
        layer2_parameters_[j] = Sigmoid(layer2_parameters_[j]);
      }
      // calculate layer 3 value
      for (int j = 0; j < num_neuron_layer3_; ++j) {
        layer3_parameters_[j] = 0;
        for (int k = 0; k <= num_neuron_layer2_; ++k) {
          layer3_parameters_[j] +=
              (layer2_parameters_[k] * weight_of_network_[1][k][j]);
        }
        layer3_parameters_[j] = Sigmoid(layer3_parameters_[j]);
      }
      // start backpropagation: calculate error and partial derivative of the
      // error with respect to weights of layer3 Refer to
      // https://en.wikipedia.org/wiki/Backpropagation
      for (int j = 0; j < num_neuron_layer3_; ++j) {
        error += pow((layer3_parameters_[j] -
                      Sigmoid(output_train_instances_[i][j])),
                     2) /
                 2;
        // partial derivative with Sigmoid could be simplified as
        // (Ooutput-Target) * Output * (1-Output)
        delta_gradient_of_network_[1][j] =
            ((layer3_parameters_[j] - Sigmoid(output_train_instances_[i][j])) *
             layer3_parameters_[j] * (1 - layer3_parameters_[j]));
      }
      // backprogapate to layer 2
      for (int j = 0; j < num_neuron_layer2_; ++j) {
        delta_gradient_of_network_[0][j] = 0;
        for (int k = 0; k < num_neuron_layer3_; ++k) {
          delta_gradient_of_network_[0][j] +=
              (delta_gradient_of_network_[1][k] * weight_of_network_[1][j][k]);
        }
        delta_gradient_of_network_[0][j] = delta_gradient_of_network_[0][j] *
                                           layer2_parameters_[j] *
                                           (1 - layer2_parameters_[2]);
      }
      // update layer 2 to 3 weight_of_network_
      for (int j = 0; j < num_neuron_layer3_; ++j) {
        for (int k = 0; k <= num_neuron_layer2_; ++k) {
          weight_delta_of_network_[1][k][j] =
              learn_rate_ * layer2_parameters_[k] *
                  delta_gradient_of_network_[1][j] +
              momentum_ * weight_delta_of_network_[1][k][j];
          weight_of_network_[1][k][j] -= weight_delta_of_network_[1][k][j];
        }
      }
      // update layer 1 to 2 weight_of_network_
      for (int j = 0; j < num_neuron_layer2_; ++j) {
        for (int k = 0; k <= num_neuron_layer1_; ++k) {
          weight_delta_of_network_[0][k][j] =
              learn_rate_ * input_train_instances_[i][k] *
                  delta_gradient_of_network_[0][j] +
              momentum_ * weight_delta_of_network_[0][k][j];
          weight_of_network_[0][k][j] -= weight_delta_of_network_[0][k][j];
        }
      }
    }
    error = error / num_train_instances_;
    deltaerror = std::abs(error - perror);
    perror = error;
    if (epoch % 100 == 0)
      std::cout << " end of epoch " << epoch << " error is " << error
                << " error delta is " << deltaerror << std::endl;
    epoch++;
  }
}

std::vector<int> ann::Predict(char *test_file, bool has_truth = 1) {
  std::vector<int> predictionResult(num_test_instances_, 0);
  // this array store our prediciton
  std::vector<int> truth(num_test_instances_, 0);
  // this array store the real result for comparison
  std::vector<double> testInput((num_neuron_layer1_ + 1), 0);
  // this array store each instance for processing

  std::ifstream testInputFile(test_file);
  if (!testInputFile) {
    std::cout << "Can't open test data file!" << std::endl;
    return predictionResult;
  }

  testInput[num_neuron_layer1_] = 1;
  std::string Buf;
  // now process each test instance
  for (int i = 0; i < num_test_instances_; ++i) {
    getline(testInputFile, Buf);
    std::stringstream lineStream(Buf);

    for (int j = 0; j < num_neuron_layer1_; ++j) {
      getline(lineStream, Buf, ',');
      testInput[j] = stod(Buf);
    }
    if (has_truth) {
      getline(lineStream, Buf, ',');
      truth[i] = stod(Buf);
    }
    predictionResult[i] = DoOnePrediction(testInput);
  }

  // calculate oeverall accuracy of our prediction
  if (has_truth) Accuracy(predictionResult, truth);

  return predictionResult;
}

int ann::DoOnePrediction(std::vector<double> &testInput) {
  // calculate layer 2 value
  for (int j = 0; j < num_neuron_layer2_; ++j) {
    layer2_parameters_[j] = 0;
    for (int k = 0; k <= num_neuron_layer1_; ++k) {
      layer2_parameters_[j] +=
          (Sigmoid(testInput[k]) * weight_of_network_[0][k][j]);
    }
    layer2_parameters_[j] = Sigmoid(layer2_parameters_[j]);
  }

  // calculate layer 3 value
  for (int j = 0; j < num_neuron_layer3_; ++j) {
    layer3_parameters_[j] = 0;
    for (int k = 0; k <= num_neuron_layer2_; ++k) {
      layer3_parameters_[j] +=
          (layer2_parameters_[k] * weight_of_network_[1][k][j]);
    }
    layer3_parameters_[j] = Sigmoid(layer3_parameters_[j]);
  }
  if (std::abs(layer3_parameters_[0] - Sigmoid(1)) >
      std::abs(layer3_parameters_[0] - Sigmoid(2))) {
    return 2;
  } else {
    return 1;
  }
}

}  // end of namespace ann
}  // namespace machinelearning