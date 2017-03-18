#ifndef TWAVE_CNN_GLOBAL_H
#define TWAVE_CNN_GLOBAL_H

//the INPUT_LENGTH refer to the length of the input vector of input layer
#define INPUT_LENGTH (116)
#define BATCH_SIZE (100)
#define LEARNING_RATE (0.001)

//the value of ACTIVATION could be changed according to the enumeration 
//variable at the end of this file
#define ACTIVATION (leaky_relu)

/*The following three macro definitions are used to check the layer's or 
network's gradient, they are mutually exclusive and only one of them should be 
defined at a time, when all of them are commented out, the main function of the 
example.c is activated*/
//#define NETWORK_GRAD_CHECK
//#define CONV_LAYER_GRAD_CHECK
//#define FC_LAYER_GRAD_CHECK

#define MAX(a,b) ( ((a)>(b)) ? (a):(b) )
#define FREE(var)  free(var);\
				   var = NULL

typedef unsigned char UCHAR;

//activation function supported until now
typedef enum
{
	relu,
	leaky_relu,
	tanH
}activation;

#endif 
