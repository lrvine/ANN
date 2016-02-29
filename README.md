# ANN
Artificial neural network implementation in C++

Usage:
```
./ann [training data file] [input test file] [configuration file] [Optional: learn rate] [Optional: momentum] 
```


Example:
```
./ann data.txt test.txt cfg.txt 
```


Training & Test data format:
```
[ attribute 1 of data 1 ]  [ attribute 2 of data 1 ] ........ [ result of data 1 ]
[ attribute 1 of data 2 ]  [ attribute 2 of data 2 ] ........ [ result of data 2 ]
[ attribute 1 of data 3 ]  [ attribute 2 of data 3 ] ........ [ result of data 3 ]
```

Configuration file format:
```
[number of training instances] [number of test instances]  [number of input attributes( number of neuron at layer 1 )] [ number of neuron at layer 2 ] [number of neuron at layer 3 ]
```

Example data set is acquired from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.html?format=&task=&att=&area=&numAtt=&numIns=&type=&sort=dateUp&view=table)
