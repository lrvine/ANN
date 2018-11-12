#include <time.h>
#include <iostream>

#include "ann.h"

int main(int argc, char** argv) {
  double learnRate = 0;
  double momentum = 0;
  double maxEpoch = 0;
  char* train_file;
  char* test_file;
  char* cfg_file;
  clock_t begin;
  clock_t end;
  double time_spent;

  if (argc < 4) {
    std::cout << " You need to provide train_fileing data, test_file data, and "
                 "configuration for prediction. Please read README"
              << std::endl;
    return -1;
  } else {
    train_file = argv[1];
    test_file = argv[2];
    cfg_file = argv[3];
  }

  if (argc >= 7) {
    maxEpoch = atof(argv[6]);
  } else {
    maxEpoch = 5000;
  }
  if (argc >= 6) {
    momentum = atof(argv[5]);
  } else {
    momentum = 0.2;
  }
  if (argc >= 5) {
    learnRate = atof(argv[4]);
  } else {
    learnRate = 0.01;
  }

  begin = clock();

  machinelearning::ann::ann aneuralnet(cfg_file, learnRate, momentum, maxEpoch);
  aneuralnet.Train(train_file);
  aneuralnet.Predict(test_file);

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  std::cout << "Time spent " << time_spent << " seconds " << std::endl;

  return 0;
}
