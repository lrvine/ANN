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
	int numlayer=3;
	double targeterror=0.001;
	double learnrate=0.9;
	double momentum=0.5;
	unsigned long long int maxepoch=32767;
	double sigmoid(double);
	double fsigmoid(double);
	//calculate the probability of each choice and choose the greatest one as our prediction
	void classifier( double *** , double *, double *, char* );  
	void accuracy(int * , int * ); // claculate the accuracy
public:
	ann( char* , char* );
};
#endif
