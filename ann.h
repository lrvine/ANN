#ifndef ann_h
#define ann_h
class ann
{
protected:
	int numberTrainInstances;  //store the number of training instances
	int numberTestInstances;   //store the number of testing instances
	double ** inputTrainInstances; //store the whole input of training instances
	double ** outputTrainInstances; //store the whole output of training instances

	// Neural Network structure
	int numberNeuronLayer1;
	int numberNeuronLayer2;
	int numberNeuronLayer3;
	int numberLayer;

	// Neural Network parameters
	double * valueLayer2;
	double * valueLayer3;
	double *** weightOfNetwork; 
	double *** weightDeltaOfNetwork;
	double ** deltaGradientOfNetwork;

	// Neural Netowrk attributes
	double targetError;
	double learnRate;
	double momentum;
	double maxWeightForInit;
	unsigned long long int maxEpoch;

	void readConfiguration( char * ); // read configuration file
	void storeTrainingData( char* ); // store training data
	void releaseTrainingData(); // store training data
	void initNetworkParameter(); // initialize the netowrk 
	void printNetworkParameter();// print out network parameter for debug
	void optimizeNetworkParameter();// optimize the network parameter via training data
	int doOnePrediction( double * ); //calculate the probability of each choice and choose the greatest one as our prediction
	void calculateAccuracy( int * , int * ); // claculate the accuracy
	double sigmoid(double);
public:
	ann( char* train_file , char* configuration_file, double learnRate=0.01 , double momentum=0.2, double maxEpoch=5000, double maxWeightForInit=0.3, double targetError=0.000000000001, int numberLayer=3 );
	~ann();
	void doClassify( char* );
};
#endif
