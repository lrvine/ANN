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
	double sigmoid(double);
	//calculate the probability of each choice and choose the greatest one as our prediction
	void classifier( double *** , double *, double *, char* );  
	void accuracy(int * , int * ); // claculate the accuracy
public:
	ann( char* train , char* test , char* configure, double learnrate=0.01 , double momentum=0.2, double maxepoch=10000, double maxwinit=0.3, double targeterror=0.000000000001, int numlayer=3 );
};
#endif
