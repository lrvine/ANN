#include <iostream>
#include "ann.h"

using namespace std;


int main( int argc, char** argv ){

double learnRate=0;
double momentum=0;
double maxEpoch=0;
char* train_file;
char* test_file;
char* cfg_file;

if( argc < 4 ){
	cout<<" You need to provide train_fileing data, test_file data, and configuration for prediction. Please read README"<<endl;
	return -1;
}else{
	train_file = argv[1];
	test_file = argv[2];
	cfg_file = argv[3];
}

if( argc >= 7 ){
	maxEpoch = atof(argv[6]);
	momentum = atof(argv[5]);
	learnRate = atof(argv[4]);
  	ann a(train_file, cfg_file, learnRate , momentum, maxEpoch);
	a.doClassify(test_file);
}else if( argc == 6 ){
	momentum = atof(argv[5]);
	learnRate = atof(argv[4]);
  	ann a(train_file, cfg_file, learnRate , momentum);
	a.doClassify(test_file);
}else if( argc == 5 ){
	learnRate = atof(argv[4]);
  	ann a(train_file, cfg_file, learnRate);
	a.doClassify(test_file);
}else if( argc == 4 ){
  	ann a(train_file, cfg_file);
	a.doClassify(test_file);
}

return 0;
}


