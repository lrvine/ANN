# ANN
Artificial neural network implementation in C++


Usage:
```
./ann [training data file] [input test file] [configuration file] [Optional: learn rate] [Optional: momentum] 
```


Example:
```
./ann breast_cancer_data/breast-cancer-wisconsin-400-records-train breast_cancer_data/breast-cancer-wisconsin-299-records-test breast_cancer_data/breast-cancer-wisconsin.cfg
```


Training & Test data are in CSV:
```
[ attribute 1 of data 1 ],[ attribute 2 of data 1 ], ........ ,[ result of data 1 ]
[ attribute 1 of data 2 ],[ attribute 2 of data 2 ], ........ ,[ result of data 2 ]
[ attribute 1 of data 3 ],[ attribute 2 of data 3 ], ........ ,[ result of data 3 ]
```

Configuration file format:
```
[number of training instances] [number of test instances]  [number of input attributes( number of neuron at layer 1 )] [ number of neuron at layer 2 ] [number of neuron at layer 3 ]
```

Data set is acquired from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html)



Performance:
When running 5000 epoch with compile option "-std=c++2a -c -Ofast -march=native -Wall" on 1.6 GHz Dual-Core Intel Core i5,
putting everything in a few big functions has the best performance, it cost around 2.68 seconds.
Breaking up the functions to align with clean code practice will slow it to 2.71 seconds.
Replace all dynamically allocated array with vectors which is the current version will further slow it to 3.52 seconds.
