#include <iostream>
#include <stdlib.h>

using namespace std;
#include "ann.h"


int main( int argc, char** argv ){

double learnrate=0;
double momentum=0;
double maxepoch=0;
char* train;
char* input;

if( argc >= 6 ){
	maxepoch = atof(argv[5]);
	momentum = atof(argv[4]);
	learnrate = atof(argv[3]);
	train = argv[1];
	input = argv[2];
  	ann a(train, input, learnrate , momentum, maxepoch);
}else if( argc >= 5 ){
	momentum = atof(argv[4]);
	learnrate = atof(argv[3]);
	train = argv[1];
	input = argv[2];
  	ann a(train, input, learnrate , momentum);
}else if( argc >= 4 ){
	learnrate = atof(argv[3]);
	train = argv[1];
	input = argv[2];
  	ann a(train, input, learnrate);
}else if( argc == 3 ){
	train = argv[1];
	input = argv[2];
  	ann a(train, input);
}else {
	cout<<" You need to provide training data and input data for prediction. Please read README"<<endl;
}


return 0;
}


