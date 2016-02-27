#ifndef ann_h
#define ann_h
class ann
{
protected:
	int traininstances;  //store the number of training instances
	int testinstances;   //store the number of testing instances
	int neulayer1;
	int neulayer2;
	int neulayer3;
	int numlayer;
	double targeterror;
	double learnrate;
	double momentum;
	double maxwinit;
	unsigned long long int maxepoch;
	double * vlayer2;
	double * vlayer3;
	double ** input;
	double ***weight; 
	double ***deltaw;
	double **deltag;
	double ** correct;

	//calculate the probability of each choice and choose the greatest one as our prediction
	int doOnePrediction( double *);
	void readConfiguration( char *); // read configuration file
	void storeTrainingData( char*); // store training data
	void initNetworkParameter(); 
	void printNetworkParameter();// print out network parameter for debug
	void optimizeNetworkParameter();// optimize the network parameter via training data
	void calculateAccuracy(int * , int * ); // claculate the accuracy
	double sigmoid(double);
public:
	ann( char* train_file , char* configuration_file, double learnrate=0.01 , double momentum=0.2, double maxepoch=5000, double maxwinit=0.3, double targeterror=0.000000000001, int numlayer=3 );
	~ann();
	void doClassify( char* );
};
#endif
