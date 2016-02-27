#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <limits>

#include "ann.h"

using namespace std;


//initialize all the information we need from training data
ann::ann( char* train_file , char* configuration_file, double ilearnrate , double imomentum, double imaxepoch, double imaxwinit, double itargeterror, int inumlayer )
{
	//set initial value
	learnrate=ilearnrate;
	momentum=imomentum;
	maxepoch=imaxepoch;
	maxwinit=imaxwinit;
	targeterror=itargeterror;
	numlayer=inumlayer;

	//read configuration
	readConfiguration(configuration_file);

	//read training data
	storeTrainingData(train_file);
	
	//allocate memory for neulayer2
	vlayer2 = new double[neulayer2+1];

	//init bais
	vlayer2[neulayer2]=1;

	//allocate memory for neulayer3
	vlayer3 = new double[neulayer3];

	//allocate memory and initialize neural layer parameters
	initNetworkParameter();		
//	printNetworkParameter();
	optimizeNetworkParameter();
//	printNetworkParameter();
}

ann::~ann()
{
	for(int i=0; i<traininstances; i++){
		delete []  input[i];
		delete []  correct[i];
	}
	delete [] input;
	delete [] correct;
	delete vlayer2;
	delete vlayer3;
	for(int i=0; i<(numlayer-1); i++){
	}
}

void ann::doClassify( char * test_file)
{

	ifstream testing(test_file);
	if(!testing){cout<<"Can't open test data file!"<<endl;return;}

	// prepare memeory space for prediciton
	int *realResult= new int[testinstances]; //this array store the real result for comparison
	if( realResult == NULL){
		cout << "Error: memory could not be allocated";
		return;
	}
	for(int w=0; w<testinstances; w++)
	{
		realResult[w]=0;
	}

	int *outcome=new int[testinstances]; //this array store our prediciton
	if( outcome == NULL){
		cout << "Error: memory could not be allocated";
		return;
	}
	for(int f=0; f<testinstances; f++)
	{
		outcome[f]=0;
	}

	double *testin=new double [neulayer1+1]; //this array store each instance for processing
	testin[neulayer1]=1;

	// now process each test instance
	for( int i=0 ; i<testinstances ; i++)
	{
		// read one instance for prediction
		for (int u=0 ; u<neulayer1; u++)
			testing>>testin[u];
		testing>>realResult[i];

		outcome[i]= doOnePrediction(testin);
	}

	// calculate oeverall accuracy of our prediction
	calculateAccuracy ( outcome , realResult );

	delete realResult;
	delete outcome;
	delete testin;

} 

int ann::doOnePrediction( double * testin )
{
	// calculate layer 2 value
	for(int j=0; j< neulayer2 ; j++){
		vlayer2[j]=0;
		for(int k=0; k<= neulayer1 ; k++){
			vlayer2[j]+=(sigmoid(testin[k])*weight[0][k][j]);					
		}
		vlayer2[j]=sigmoid(vlayer2[j]);
	}

	// calculate layer 3 value
	for(int j=0; j< neulayer3 ; j++){
		vlayer3[j]=0;
		for(int k=0; k<= neulayer2 ; k++){
			vlayer3[j]+=(vlayer2[k]*weight[1][k][j]);	
		}
		vlayer3[j]=sigmoid(vlayer3[j]);
	//	cout<<"test pattern "<<i<<" ouptut "<<vlayer3[j]<<endl;
	}
	if( abs(vlayer3[0]-sigmoid(1)) > abs(vlayer3[0]-sigmoid(2)) ){
		return 2;
	}else{
		return 1;
	}
}

void ann::readConfiguration( char* configuration_file )
{
	//read configuration 
	ifstream cfg;
        cfg.open(configuration_file);
        if(!cfg){cout<<"Can't open configure file!"<<endl;return;}

	cfg>>traininstances>>testinstances>>neulayer1>>neulayer2>>neulayer3;
	cfg.close();
	//cout<<traininstances<<testinstances<<neulayer1<<neulayer2<<neulayer3;
}

void ann::storeTrainingData( char * train_file )
{
	ifstream training;
        training.open(train_file);
        if(!training){cout<<"Can't open training data file!"<<endl;return;}
    

	//allocate memory for input data 
	input= new (nothrow) double *[traininstances];  
	if( input == NULL){
		cout << "Error: memory could not be allocated";
		return;
	}
	for(int b=0; b<traininstances; b++){   
		input[b]= new double[neulayer1+1];
		if( input[b] == NULL){
			cout << "Error: memory could not be allocated";
			return;
		}
	}

	//allocate memory for correct output 
	correct= new (nothrow) double *[traininstances];  
	if( correct == NULL){
		cout << "Error: memory could not be allocated";
		return;
	}
	for(int b=0; b<traininstances; b++){   
		correct[b]= new double[neulayer3];
		if( correct[b] == NULL){
			cout << "Error: memory could not be allocated";
			return;
		}
	}

	// store input data and correct output
	for(int i=0; i<traininstances; i++){
		for(int j=0; j<neulayer1; j++){
			training>>input[i][j];
		}
		for(int j=0; j<neulayer3; j++){
			training>>correct[i][j];
		}
		//init bias
		input[i][neulayer1]=1;
	}

}

void ann::initNetworkParameter()
{
	random_device rd;
	mt19937 mt(rd());
	uniform_real_distribution<double> dist(0, 1);

	weight = new double **[numlayer-1];
	deltaw = new double **[numlayer-1];
	deltag = new double *[numlayer-1];
	for(int i=0; i<(numlayer-1); i++){
		if(i==0){
			weight[i] = new double *[neulayer1+1];
			deltaw[i] = new double *[neulayer1+1];
			deltag[i] = new double[neulayer2];
			for(int j=0; j<neulayer2; j++){
				deltag[i][j]=0;
			}
			for(int j=0; j<=neulayer1; j++){
				weight[i][j] = new double [neulayer2+1];
				deltaw[i][j] = new double [neulayer2+1];
				for(int k=0; k<neulayer2; k++){
					weight[i][j][k]=( 2*(dist(mt)-0.5)*maxwinit );
					deltaw[i][j][k]=0;
				}
			}
		}
		if(i==1){
			weight[i] = new double *[neulayer2+1];
			deltaw[i] = new double *[neulayer2+1];
			deltag[i] = new double [neulayer3];
			for(int j=0; j<neulayer3; j++){
				deltag[i][j]=0;
			}
			for(int j=0; j<=neulayer2; j++){
				weight[i][j] = new double [neulayer3];
				deltaw[i][j] = new double [neulayer3];
				for(int k=0; k<neulayer3; k++){
					weight[i][j][k]=( 2*(dist(mt)-0.5)*maxwinit );
					deltaw[i][j][k]=0;
				}
			}
		}
	}
}

void ann::printNetworkParameter()
{
	for(int j=0; j<=neulayer1; j++){
		for(int k=0; k< neulayer2; k++){
		  cout<<weight[0][j][k]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	for(int j=0; j<=neulayer2; j++){
		for(int k=0; k< neulayer3; k++){
		  cout<<weight[1][j][k]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
}

void ann::optimizeNetworkParameter()
{	
	/*
	// shuffle input pattern order  : This didn't seems to improve prediction rate. Comment out for now.
	srand ( unsigned ( std::time(0) ) );
	vector<int> ilist;
	for(int i=0; i<traininstances ; i++){
		ilist.push_back(i);
	}
	*/
	unsigned long long int epoch=0;
	double error = numeric_limits<double>::max();
	double perror = numeric_limits<double>::max();
	double deltaerror = numeric_limits<double>::max();
	while( epoch < maxepoch && deltaerror > targeterror ){
		// train each pattern for one epoch
		error=0;
//		random_shuffle(ilist.begin(),ilist.end());
		for(int i=0; i< traininstances ; i++){
			// calculate layer 2 value
			for(int j=0; j< neulayer2 ; j++){
				vlayer2[j]=0;
				for(int k=0; k<= neulayer1 ; k++){
					vlayer2[j]+=(sigmoid(input[i][k])*weight[0][k][j]);	
				}
				vlayer2[j]=sigmoid(vlayer2[j]);
		//		cout<<vlayer2[j]<<" ";
			}
		//	cout<<endl;
			// calculate layer 3 value
			for(int j=0; j< neulayer3 ; j++){
				vlayer3[j]=0;
				for(int k=0; k<= neulayer2 ; k++){
					vlayer3[j]+=(vlayer2[k]*weight[1][k][j]);	
				}
				vlayer3[j]=sigmoid(vlayer3[j]);
				error += pow((vlayer3[j]-sigmoid(correct[i][j])),2)/2;
				//cout<<"vlayer3[j] "<<vlayer3[j]<<" correct "<<sigmoid(correct[i][j])<<"error "<<error<<endl;
				// calculate delta gradient of layer3
				deltag[1][j]=((vlayer3[j]-sigmoid(correct[i][j]))*vlayer3[j]*(1-vlayer3[j]));
			}
			// calculate delta gradient of layer2
			for(int j=0; j< neulayer2 ; j++){
				deltag[0][j]=0;
				for(int k=0; k< neulayer3 ; k++){
					deltag[0][j]+=(deltag[1][k]*weight[1][j][k]);	
				}
				deltag[0][j]=deltag[0][j]*vlayer2[j]*(1-vlayer2[2]);
			}
			// update layer 2 to 3 weight
			for(int j=0; j< neulayer3 ; j++){
				for(int k=0; k<= neulayer2 ; k++){
					deltaw[1][k][j]= learnrate*vlayer2[k]*deltag[1][j]+ momentum*deltaw[1][k][j];	
					weight[1][k][j]-=deltaw[1][k][j];
				}
			}
			// update layer 1 to 2 weight
			for(int j=0; j< neulayer2 ; j++){
				for(int k=0; k<= neulayer1 ; k++){
					deltaw[0][k][j]= learnrate*input[i][k]*deltag[0][j]+ momentum*deltaw[0][k][j];	
					weight[0][k][j]-=deltaw[0][k][j];
				}
			}

		}
		error=error/traininstances;
		deltaerror=abs(error-perror);
		perror=error;
		cout<<" end of epoch "<<epoch<<" error is "<<error<<" error delta is "<<deltaerror<<endl;
		epoch++;
	}

}

void ann::calculateAccuracy(int *outcome , int * result)
{

	double correct=0;// store the number of correct predictions

	for( int i=0 ; i<testinstances; i++)//count the number of correct predictions 
	{
		if (outcome[i]==result[i])
			correct++;
		cout<<"predict to be "<<outcome[i]<<" is actually "<<result[i]<<endl;
	}
	
	cout<<"total "<<testinstances<<" data have "<<correct<<" correct prediction"<< endl;

	double percentage=correct/testinstances; // calculate the accuracy

	cout<<"accuracy is "<<percentage*100<<"%"<<endl;
}

double inline ann::sigmoid(double x )
{
	x=exp(-x);
	x=1/(1+x);
	return x;
}
