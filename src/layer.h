
#ifndef TWAVE_CNN_LAYER_H
#define TWAVE_CNN_LAYER_H

#include "global.h"

typedef struct conv_layer {
	int input_length;
	int scale;
	int core_num;
	int core_length;
	int output_node_num;
	int output_length;
	double *weight;
	double *bias;
	double *output;
	double *sensitivity;
	double *gradient;
	double *bias_grad;
}CONV_LAYER;

typedef struct input_layer {
	int batch_size;
	int input_length;
	int train_sample_num;
	int test_sample_num;
	double *train_data;
	double *test_data;
}INPUT_LAYER;

typedef struct pooling_layer {
	int invect_length;
	int invect_num;
	int outvect_length;
	int outnode_num;
	int *max_index;
	double *output;
	double *sensitivity;
}POOLING_LAYER;

typedef struct fc_layer {
	int invect_length;
	int outvect_length;
	double *weight;
	double bias;
	double bias_grad;
	double *output;
	double *sensitivity;
	double *gradient;
	UCHAR *train_label;
	UCHAR *test_label;
}FC_LAYER;

#endif 

