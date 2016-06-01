#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <limits>

#include "ann.h"

using namespace std;


//initialize all the information we need from training data
ann::ann( char* train_file , char* configuration_file, double ilearnRate , double imomentum, double imaxEpoch, double imaxWeightForInit, double itargetError, int inumberLayer )
{
	//set initial value
	learnRate=ilearnRate;
	momentum=imomentum;
	maxEpoch=imaxEpoch;
	maxWeightForInit=imaxWeightForInit;
	targetError=itargetError;
	numberLayer=inumberLayer;

	//read configuration
	readConfiguration(configuration_file);

	//allocate memory for training data
	allocateMemoryForTrainingData();
	//read training data
	storeTrainingData(train_file);
	
	//allocate memory for numberNeuronLayer2
	valueLayer2 = new double[numberNeuronLayer2+1];

	//init bais
	valueLayer2[numberNeuronLayer2]=1;

	//allocate memory for numberNeuronLayer3
	valueLayer3 = new double[numberNeuronLayer3];

	//allocate memory and initialize neural layer parameters
	initNetworkParameter();		
//	printNetworkParameter();
	optimizeNetworkParameter();
	releaseTrainingData();
//	printNetworkParameter();
}

ann::~ann()
{
	delete [] valueLayer2;
	delete [] valueLayer3;

	for(int i=0; i<(numberLayer-1); i++){
		for(int j=0; j<= numberNeuronLayer1; j++){
			delete [] weightOfNetwork[i][j];
			delete [] weightDeltaOfNetwork[i][j];
		}
		delete [] weightOfNetwork[i];
		delete [] weightDeltaOfNetwork[i];
		delete [] deltaGradientOfNetwork[i];
	}
	delete [] weightOfNetwork;
	delete [] weightDeltaOfNetwork;
	delete [] deltaGradientOfNetwork;
}


void ann::readConfiguration( char* configuration_file )
{
	//read configuration 
	ifstream cfg;
        cfg.open(configuration_file);
        if(!cfg){cout<<"Can't open configure file!"<<endl;return;}

	cfg>>numberTrainInstances>>numberTestInstances>>numberNeuronLayer1>>numberNeuronLayer2>>numberNeuronLayer3;
	cfg.close();
	//cout<<numberTrainInstances<<numberTestInstances<<numberNeuronLayer1<<numberNeuronLayer2<<numberNeuronLayer3;
}

void ann::allocateMemoryForTrainingData()
{
	//allocate memory for inputTrainInstances data 
	inputTrainInstances= new (nothrow) double *[numberTrainInstances];  
	if( inputTrainInstances == NULL){
		cout << "Error: memory could not be allocated";
		return;
	}
	for(int b=0; b<numberTrainInstances; b++){   
		inputTrainInstances[b]= new double[numberNeuronLayer1+1];
		if( inputTrainInstances[b] == NULL){
			cout << "Error: memory could not be allocated";
			return;
		}
	}

	//allocate memory for outputTrainInstances output 
	outputTrainInstances= new (nothrow) double *[numberTrainInstances];  
	if( outputTrainInstances == NULL){
		cout << "Error: memory could not be allocated";
		return;
	}
	for(int b=0; b<numberTrainInstances; b++){   
		outputTrainInstances[b]= new double[numberNeuronLayer3];
		if( outputTrainInstances[b] == NULL){
			cout << "Error: memory could not be allocated";
			return;
		}
	}

}
void ann::storeTrainingData( char * train_file )
{
	ifstream training;
        training.open(train_file);
        if(!training){cout<<"Can't open training data file!"<<endl;return;}

	// store inputTrainInstances data and outputTrainInstances output
	for(int i=0; i<numberTrainInstances; i++){
		for(int j=0; j<numberNeuronLayer1; j++){
			training>>inputTrainInstances[i][j];
		}
		for(int j=0; j<numberNeuronLayer3; j++){
			training>>outputTrainInstances[i][j];
		}
		//init bias
		inputTrainInstances[i][numberNeuronLayer1]=1;
	}
}

void ann::releaseTrainingData()
{
	for(int i=0; i<numberTrainInstances; i++){
		delete []  inputTrainInstances[i];
		delete []  outputTrainInstances[i];
	}
	delete [] inputTrainInstances;
	delete [] outputTrainInstances;
}

void ann::initNetworkParameter()
{
	random_device rd;
	mt19937 mt(rd());
	uniform_real_distribution<double> dist(0, 1);

	weightOfNetwork = new double **[numberLayer-1];
	weightDeltaOfNetwork = new double **[numberLayer-1];
	deltaGradientOfNetwork = new double *[numberLayer-1];
	for(int i=0; i<(numberLayer-1); i++){
		if(i==0){
			weightOfNetwork[i] = new double *[numberNeuronLayer1+1];
			weightDeltaOfNetwork[i] = new double *[numberNeuronLayer1+1];
			deltaGradientOfNetwork[i] = new double[numberNeuronLayer2];
			for(int j=0; j<numberNeuronLayer2; j++){
				deltaGradientOfNetwork[i][j]=0;
			}
			for(int j=0; j<=numberNeuronLayer1; j++){
				weightOfNetwork[i][j] = new double [numberNeuronLayer2+1];
				weightDeltaOfNetwork[i][j] = new double [numberNeuronLayer2+1];
				for(int k=0; k<numberNeuronLayer2; k++){
					weightOfNetwork[i][j][k]=( 2*(dist(mt)-0.5)*maxWeightForInit );
					weightDeltaOfNetwork[i][j][k]=0;
				}
			}
		}
		if(i==1){
			weightOfNetwork[i] = new double *[numberNeuronLayer2+1];
			weightDeltaOfNetwork[i] = new double *[numberNeuronLayer2+1];
			deltaGradientOfNetwork[i] = new double [numberNeuronLayer3];
			for(int j=0; j<numberNeuronLayer3; j++){
				deltaGradientOfNetwork[i][j]=0;
			}
			for(int j=0; j<=numberNeuronLayer2; j++){
				weightOfNetwork[i][j] = new double [numberNeuronLayer3];
				weightDeltaOfNetwork[i][j] = new double [numberNeuronLayer3];
				for(int k=0; k<numberNeuronLayer3; k++){
					weightOfNetwork[i][j][k]=( 2*(dist(mt)-0.5)*maxWeightForInit );
					weightDeltaOfNetwork[i][j][k]=0;
				}
			}
		}
	}
}

void ann::printNetworkParameter()
{
	for(int j=0; j<=numberNeuronLayer1; j++){
		for(int k=0; k< numberNeuronLayer2; k++){
		  cout<<weightOfNetwork[0][j][k]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	for(int j=0; j<=numberNeuronLayer2; j++){
		for(int k=0; k< numberNeuronLayer3; k++){
		  cout<<weightOfNetwork[1][j][k]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
}

void ann::optimizeNetworkParameter()
{	
	/*
	// shuffle inputTrainInstances pattern order  : This didn't seems to improve prediction rate. Comment out for now.
	srand ( unsigned ( std::time(0) ) );
	vector<int> ilist;
	for(int i=0; i<numberTrainInstances ; i++){
		ilist.push_back(i);
	}
	*/
	unsigned long long int epoch=0;
	double error = numeric_limits<double>::max();
	double perror = numeric_limits<double>::max();
	double deltaerror = numeric_limits<double>::max();
	while( epoch < maxEpoch && deltaerror > targetError ){
		// train each pattern for one epoch
		error=0;
//		random_shuffle(ilist.begin(),ilist.end());
		for(int i=0; i< numberTrainInstances ; i++){
			// calculate layer 2 value
			for(int j=0; j< numberNeuronLayer2 ; j++){
				valueLayer2[j]=0;
				for(int k=0; k<= numberNeuronLayer1 ; k++){
					valueLayer2[j]+=(sigmoid(inputTrainInstances[i][k])*weightOfNetwork[0][k][j]);	
				}
				valueLayer2[j]=sigmoid(valueLayer2[j]);
		//		cout<<valueLayer2[j]<<" ";
			}
		//	cout<<endl;
			// calculate layer 3 value
			for(int j=0; j< numberNeuronLayer3 ; j++){
				valueLayer3[j]=0;
				for(int k=0; k<= numberNeuronLayer2 ; k++){
					valueLayer3[j]+=(valueLayer2[k]*weightOfNetwork[1][k][j]);	
				}
				valueLayer3[j]=sigmoid(valueLayer3[j]);
				error += pow((valueLayer3[j]-sigmoid(outputTrainInstances[i][j])),2)/2;
				//cout<<"valueLayer3[j] "<<valueLayer3[j]<<" outputTrainInstances "<<sigmoid(outputTrainInstances[i][j])<<"error "<<error<<endl;
				// calculate delta gradient of layer3
				deltaGradientOfNetwork[1][j]=((valueLayer3[j]-sigmoid(outputTrainInstances[i][j]))*valueLayer3[j]*(1-valueLayer3[j]));
			}
			// calculate delta gradient of layer2
			for(int j=0; j< numberNeuronLayer2 ; j++){
				deltaGradientOfNetwork[0][j]=0;
				for(int k=0; k< numberNeuronLayer3 ; k++){
					deltaGradientOfNetwork[0][j]+=(deltaGradientOfNetwork[1][k]*weightOfNetwork[1][j][k]);	
				}
				deltaGradientOfNetwork[0][j]=deltaGradientOfNetwork[0][j]*valueLayer2[j]*(1-valueLayer2[2]);
			}
			// update layer 2 to 3 weightOfNetwork
			for(int j=0; j< numberNeuronLayer3 ; j++){
				for(int k=0; k<= numberNeuronLayer2 ; k++){
					weightDeltaOfNetwork[1][k][j]= learnRate*valueLayer2[k]*deltaGradientOfNetwork[1][j]+ momentum*weightDeltaOfNetwork[1][k][j];	
					weightOfNetwork[1][k][j]-=weightDeltaOfNetwork[1][k][j];
				}
			}
			// update layer 1 to 2 weightOfNetwork
			for(int j=0; j< numberNeuronLayer2 ; j++){
				for(int k=0; k<= numberNeuronLayer1 ; k++){
					weightDeltaOfNetwork[0][k][j]= learnRate*inputTrainInstances[i][k]*deltaGradientOfNetwork[0][j]+ momentum*weightDeltaOfNetwork[0][k][j];	
					weightOfNetwork[0][k][j]-=weightDeltaOfNetwork[0][k][j];
				}
			}

		}
		error=error/numberTrainInstances;
		deltaerror=abs(error-perror);
		perror=error;
		cout<<" end of epoch "<<epoch<<" error is "<<error<<" error delta is "<<deltaerror<<endl;
		epoch++;
	}

}

void ann::doClassify( char * test_file)
{

	ifstream testInputg(test_file);
	if(!testInputg){cout<<"Can't open test data file!"<<endl;return;}

	// prepare memeory space for prediciton
	int *realResult= new int[numberTestInstances]; //this array store the real result for comparison
	if( realResult == NULL){
		cout << "Error: memory could not be allocated";
		return;
	}
	for(int w=0; w<numberTestInstances; w++)
	{
		realResult[w]=0;
	}

	int *predictionResult=new int[numberTestInstances]; //this array store our prediciton
	if( predictionResult == NULL){
		cout << "Error: memory could not be allocated";
		return;
	}
	for(int f=0; f<numberTestInstances; f++)
	{
		predictionResult[f]=0;
	}

	double *testInput=new double [numberNeuronLayer1+1]; //this array store each instance for processing
	testInput[numberNeuronLayer1]=1;

	// now process each test instance
	for( int i=0 ; i<numberTestInstances ; i++)
	{
		// read one instance for prediction
		for (int u=0 ; u<numberNeuronLayer1; u++)
			testInputg>>testInput[u];
		testInputg>>realResult[i];

		predictionResult[i]= doOnePrediction(testInput);
	}

	// calculate oeverall accuracy of our prediction
	calculateAccuracy ( predictionResult , realResult );

	delete [] realResult;
	delete [] predictionResult;
	delete [] testInput;

} 

int ann::doOnePrediction( double * testInput )
{
	// calculate layer 2 value
	for(int j=0; j< numberNeuronLayer2 ; j++){
		valueLayer2[j]=0;
		for(int k=0; k<= numberNeuronLayer1 ; k++){
			valueLayer2[j]+=(sigmoid(testInput[k])*weightOfNetwork[0][k][j]);					
		}
		valueLayer2[j]=sigmoid(valueLayer2[j]);
	}

	// calculate layer 3 value
	for(int j=0; j< numberNeuronLayer3 ; j++){
		valueLayer3[j]=0;
		for(int k=0; k<= numberNeuronLayer2 ; k++){
			valueLayer3[j]+=(valueLayer2[k]*weightOfNetwork[1][k][j]);	
		}
		valueLayer3[j]=sigmoid(valueLayer3[j]);
	//	cout<<"test pattern "<<i<<" ouptut "<<valueLayer3[j]<<endl;
	}
	if( abs(valueLayer3[0]-sigmoid(1)) > abs(valueLayer3[0]-sigmoid(2)) ){
		return 2;
	}else{
		return 1;
	}
}

void ann::calculateAccuracy(int *predictionResult , int * result)
{

	double outputTrainInstances=0;// store the number of outputTrainInstances predictions

	for( int i=0 ; i<numberTestInstances; i++)//count the number of outputTrainInstances predictions 
	{
		if (predictionResult[i]==result[i])
			outputTrainInstances++;
		cout<<"predict to be "<<predictionResult[i]<<" is actually "<<result[i]<<endl;
	}
	
	cout<<"Total "<<numberTestInstances<<" test datas have "<<outputTrainInstances<<" correct predictions"<< endl;

	double percentage=outputTrainInstances/numberTestInstances; // calculate the accuracy

	cout<<"accuracy is "<<percentage*100<<"%"<<endl;
}

double inline ann::sigmoid(double x )
{
	x=exp(-x);
	x=1/(1+x);
	return x;
}
