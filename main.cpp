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
char* cfg;

if( argc >= 7 ){
	maxepoch = atof(argv[6]);
	momentum = atof(argv[5]);
	learnrate = atof(argv[4]);
	train = argv[1];
	input = argv[2];
	cfg = argv[3];
  	ann a(train, input, cfg, learnrate , momentum, maxepoch);
}else if( argc >= 6 ){
	momentum = atof(argv[5]);
	learnrate = atof(argv[4]);
	train = argv[1];
	input = argv[2];
	cfg = argv[3];
  	ann a(train, input, cfg, learnrate , momentum);
}else if( argc >= 5 ){
	learnrate = atof(argv[4]);
	train = argv[1];
	input = argv[2];
	cfg = argv[3];
  	ann a(train, input, cfg, learnrate);
}else if( argc == 4 ){
	train = argv[1];
	input = argv[2];
	cfg = argv[3];
	cout<<"ok"<<endl;
  	ann a(train, input, cfg);
}else {
	cout<<" You need to provide training data and input data for prediction. Please read README"<<endl;
}


return 0;
}


