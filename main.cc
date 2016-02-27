#include <iostream>
#include <stdlib.h>
#include "ann.h"

using namespace std;


int main( int argc, char** argv ){

double learnrate=0;
double momentum=0;
double maxepoch=0;
char* train;
char* input;
char* cfg;

if( argc < 4 ){
	cout<<" You need to provide training data, input data, and configuration for prediction. Please read README"<<endl;
	return -1;
}else{
	train = argv[1];
	input = argv[2];
	cfg = argv[3];
}

if( argc >= 7 ){
	maxepoch = atof(argv[6]);
	momentum = atof(argv[5]);
	learnrate = atof(argv[4]);
  	ann a(train, cfg, learnrate , momentum, maxepoch);
	a.doClassify(input);
}else if( argc == 6 ){
	momentum = atof(argv[5]);
	learnrate = atof(argv[4]);
  	ann a(train, cfg, learnrate , momentum);
	a.doClassify(input);
}else if( argc == 5 ){
	learnrate = atof(argv[4]);
  	ann a(train, cfg, learnrate);
	a.doClassify(input);
}else if( argc == 4 ){
  	ann a(train, cfg);
	a.doClassify(input);
}

return 0;
}


