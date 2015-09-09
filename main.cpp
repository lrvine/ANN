#include <iostream>
#include <stdlib.h>

using namespace std;
#include "ann.h"


int main( int argc, char** argv ){

int method=0;
char* train;
char* input;

if( argc >= 4 ){
	method = atoi(argv[3]);
	train = argv[1];
	input = argv[2];
}else if( argc == 3 ){
	train = argv[1];
	input = argv[2];
	cout<<" use default method"<<endl;
}else {
	cout<<" You need to provide training data and input data for prediction. Please read README"<<endl;
}

if( method == 0 ){
  ann a(train, input);
}
return 0;
}


