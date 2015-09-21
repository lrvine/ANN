# ANN
Artificial neural network implementation in C++

Usage:

ann [training data file] [input test file] [configuration file] [Optional: learn rate] [Optional: momentum] 



Example:

./ann data.txt test.txt cfg.txt 



Training & Test data format:

[ attribute 1 of data 1 ]  [ attribute 2 of data 1 ] ........ [ result of data 1 ]
[ attribute 1 of data 2 ]  [ attribute 2 of data 2 ] ........ [ result of data 2 ]
[ attribute 1 of data 3 ]  [ attribute 2 of data 3 ] ........ [ result of data 3 ]


Configuration file format:

[number of training instance] [number of test instance]  [number of input attributes( number of neuron of layer 1 )] [ number of neuron of layer 2 ] [number of neuron of layer 3 ]
