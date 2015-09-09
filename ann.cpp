#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <limits>

#include "ann.h"

using namespace std;


//initialize all the information we need from training data
ann::ann( char * train, char* test )
{
	ifstream training;
        training.open(train);
        if(!training){cout<<"Can't open training data file!"<<endl;return;}
    
	training>>traininstances>>neulayer1; // read the number of training instances and attributes

	training>>neulayer2>>neulayer3;

	//store the whole input & bias  
	double **input= new double *[traininstances];  
	for(int b=0; b<traininstances; b++)   
		input[b]= new double[neulayer1+1];


	//store the whole correct output 
	double **correct= new double *[traininstances];  
	for(int b=0; b<traininstances; b++)   
		correct[b]= new double[neulayer3];

	for(int i=0; i<traininstances; i++){
		for(int j=0; j<neulayer1; j++){
			training>>input[i][j];
		}
		for(int j=0; j<neulayer3; j++){
			training>>correct[i][j];
		}
		input[i][neulayer1]=-1;
	}
	//store value of neulayer2
	double * vlayer2 = new double[neulayer2+1];
	vlayer2[neulayer2]=-1;
	//store value of neulayer3
	double * vlayer3 = new double[neulayer3];


	random_device rd;
	mt19937 mt(rd());
	uniform_real_distribution<double> dist(0, 1);

	double ***weight = new double **[numlayer-1];
	double ***deltaw = new double **[numlayer-1];
	double **deltag = new double *[numlayer-1];
	for(int i=0; i<numlayer; i++){
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
					weight[i][j][k]=( 2*(dist(mt)-0.5) );
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
					weight[i][j][k]=( 2*(dist(mt)-0.5) );
					deltaw[i][j][k]=0;
				}
			}
		}
	}
	

	unsigned long long int epoch=0;
	double error = numeric_limits<double>::max();
	cout<<"begining of epoch "<<epoch<<" error is "<<error<<" "<<neulayer3<<" "<<neulayer2<<" "<<neulayer1<<endl;
	maxepoch=10000;
	while( epoch < maxepoch && error > targeterror ){
		// train each pattern for one epoch
		error=0;
		for(int i=0; i< traininstances ; i++){
			// calculate layer 2 value
			for(int j=0; j< neulayer2 ; j++){
				for(int k=0; k<= neulayer1 ; k++){
					vlayer2[j]+=(input[i][k]*weight[0][k][j]);	
				}
				vlayer2[j]=sigmoid(vlayer2[j]);
		//		cout<<vlayer2[j]<<" ";
			}
		//	cout<<endl;
			// calculate layer 3 value
			for(int j=0; j< neulayer3 ; j++){
				for(int k=0; k<= neulayer2 ; k++){
					vlayer3[j]+=(vlayer2[k]*weight[1][k][j]);	
				}
				vlayer3[j]=sigmoid(vlayer3[j]);
				error += pow((vlayer3[j]-correct[i][j]),2)/2;
				// calculate delta gradient of layer3
				deltag[1][j]=((vlayer3[j]-correct[i][j])*vlayer3[j]*(1-vlayer3[j]));
			}
			// calculate delta gradient of layer2
			for(int j=0; j< neulayer2 ; j++){
				for(int k=0; k< neulayer3 ; k++){
					deltag[0][j]+=(vlayer3[k]*weight[1][j][k]);	
				}
				deltag[0][j]=deltag[0][j]*vlayer2[j]*(1-vlayer2[2]);
			}
			// update layer 2 to 3 weight
			for(int j=0; j< neulayer3 ; j++){
				for(int k=0; k<= neulayer2 ; k++){
					deltaw[1][k][j]= learnrate*vlayer2[k]*deltag[1][j]+ momentum*deltaw[1][k][j];	
					weight[1][k][j]+=deltaw[1][k][j];
				}
			}
			// update layer 1 to 2 weight
			for(int j=0; j< neulayer2 ; j++){
				for(int k=0; k<= neulayer1 ; k++){
					deltaw[0][k][j]= learnrate*input[i][k]*deltag[0][j]+ momentum*deltaw[0][k][j];	
					weight[0][k][j]+=deltaw[0][k][j];
				}
			}

		}
		cout<<" end of epoch "<<epoch<<" error is "<<error<<endl;
		epoch++;
	}
	
	classifier( weight, vlayer2, vlayer3, test);	

	for(int i=0; i<traininstances; i++)
		delete []  input[i];
	delete [] input;


}
void ann::classifier( double *** weight , double * vlayer2, double * vlayer3, char * test){

	ifstream testing(test);
	if(!testing){cout<<"Can't open training data file!"<<endl;return;}

	testing>>testinstances;              //read the number of testing data

	int *result= new int[testinstances]; //this array store the real result for comparison
	for(int w=0; w<testinstances; w++)
	{
		result[w]=0;
	}

	int *outcome=new int[testinstances]; //this array store our prediciton
	for(int f=0; f<testinstances; f++)
	{
		outcome[f]=0;
	}

	double *testin=new double [neulayer1+1]; //store each instance for processing
	testin[neulayer1]=-1;

	double decision=0;; 
	for( int i=0 ; i<testinstances ; i++)
	{
		for (int u=0 ; u<neulayer1; u++)
			testing>>testin[u];
		testing>>result[i];
		// read one instance for prediction
		// calculate layer 2 value
		for(int j=0; j< neulayer2 ; j++){
			for(int k=0; k<= neulayer1 ; k++){
				vlayer2[j]+=(testin[k]*weight[0][k][j]);					
			}
			vlayer2[j]=sigmoid(vlayer2[j]);
		}
		// calculate layer 3 value
		for(int j=0; j< neulayer3 ; j++){
			for(int k=0; k<= neulayer2 ; k++){
				vlayer3[j]+=(vlayer2[k]*weight[1][k][j]);	
			}
			vlayer3[j]=sigmoid(vlayer3[j]);
			cout<<"test pattern "<<i<<" ouptut "<<vlayer3[j]<<endl;
		}
		decision=floor(vlayer3[0]+0.5);
		outcome[i]=decision;
	}
	accuracy ( outcome , result );

} 

void ann::accuracy(int *outcome , int * result)
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

double ann::sigmoid(double x ){

x=exp(-x);
x=1/(1+x);

return x;
}

double ann::fsigmoid(double x ){

x=x/(1+abs(x));

return x;
}
