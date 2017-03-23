/*
  Copyright (c) 2017, Jiemeng Zou
  All rights reserved.
  Use of this source code is governed by a BSD-style license that can be found
  in the LICENSE file.
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "global.h"
#include "layer.h"
#include "activation_function.h"

//get samples data from .tcnn file (can be customized)
int get_samples_data(double **data, int length, const char * data_file)
{
	FILE *fp;
	int total_num;
	fp = fopen(data_file, "rb");
	if (fp == NULL)
	{
		printf("open file error!\n");
		return -1;
	}
	fread(&total_num, sizeof(int), 1, fp);
	*data = (double *)calloc(total_num*length, sizeof(double));
	fread(*data, sizeof(double), total_num*length, fp);
	fclose(fp);
	return total_num;
}

//get samples label from .tcnn file (can be customized)
int get_samples_label(UCHAR **label, const char * label_file) 
{
	FILE *fp;
	int total_num;
	fp = fopen(label_file, "rb");
	if (fp == NULL)
	{
		printf("open file error!\n");
		return -1;
	}
	fread(&total_num, sizeof(int), 1, fp);
	*label = (UCHAR *)calloc(total_num, sizeof(UCHAR));
	fread(*label, sizeof(UCHAR), total_num, fp);
	fclose(fp);
	return total_num;
}

//Initialize the input layer
void initial_input_layer(INPUT_LAYER *input, const char *train_file, 
						 const char *test_file) 
{
	input->input_length = INPUT_LENGTH;
	input->batch_size = BATCH_SIZE;
	input->train_sample_num = get_samples_data(&input->train_data,
		input->input_length,train_file);
	input->test_sample_num = get_samples_data(&input->test_data,
		input->input_length,test_file);
}

//Release the memory of the input layer
void free_input_layer(INPUT_LAYER *input)
{
	FREE(input->train_data);
	FREE(input->test_data);
}

//Initialize the convolution layer
void initial_conv_layer(CONV_LAYER *conv, int input_length, int core_length, 
	int scale, int core_num)
{ 
	int i;
	conv->input_length = input_length;
	conv->core_length = core_length;
	conv->scale = scale;
	conv->core_num = core_num;
	conv->output_length = input_length - core_length + 1;
	conv->output_node_num = core_num * (input_length - core_length + 1);
	conv->bias = (double *)calloc(core_num,sizeof(double));
	conv->bias_grad = (double *)calloc(core_num, sizeof(double));
	conv->weight = (double *)calloc(scale*core_length*core_num, sizeof(double));
	conv->gradient = (double *)calloc(scale*core_length*core_num, 
									  sizeof(double));
	conv->output = (double *)calloc(core_num * (input_length - core_length + 1),
									sizeof(double));
	conv->sensitivity = (double *)calloc(core_num * (input_length -
										 core_length + 1), sizeof(double));
	//Initialize the weight with a random value between [0,0.05]
	srand((unsigned)time(NULL));
	for (i = 0; i < scale*core_length*core_num; i++)
	{
		conv->weight[i] = (0.05*rand() / RAND_MAX);
	}
}

//Release the memory of the convolution layer
void free_conv_layer(CONV_LAYER *conv)
{
	FREE(conv->weight);
	FREE(conv->bias);
	FREE(conv->output);
	FREE(conv->sensitivity);
	FREE(conv->gradient);
	FREE(conv->bias_grad);
}

//Initialize the pooling layer
void initial_pooling_layer(POOLING_LAYER *pooling, int invect_length, 
	int invect_num)
{
	pooling->invect_length = invect_length;
	pooling->invect_num = invect_num;
	pooling->outvect_length = invect_length / 2;
	pooling->outnode_num = invect_length / 2 * invect_num;
	pooling->output = (double *)calloc(invect_length / 2 * invect_num, 
		sizeof(double));
	pooling->max_index = (int *)calloc(invect_length / 2 * invect_num,
		sizeof(int));
	pooling->sensitivity = (double *)calloc(invect_length / 2 * invect_num,
		sizeof(double));

}

//Release the memory of the pooling layer
void free_pooling_layer(POOLING_LAYER *pooling)
{
	FREE(pooling->max_index);
	FREE(pooling->output);
	FREE(pooling->sensitivity);
	
}

//Initialize the fully-connected layer
void initial_fc_layer(FC_LAYER *fc, int invect_length, int outvect_length,
					  const char *train_file, const char *test_file)
{
	int i;
	fc->invect_length = invect_length;
	fc->outvect_length = outvect_length;
	fc->bias = 0.0;
	fc->bias_grad = 0.0;
	get_samples_label(&fc->train_label, train_file);
	get_samples_label(&fc->test_label, test_file);
	fc->weight = (double *)calloc(invect_length*outvect_length, sizeof(double));
	fc->gradient = (double *)calloc(invect_length*outvect_length, 
								    sizeof(double));
	fc->output = (double *)calloc(outvect_length, sizeof(double));
	fc->sensitivity = (double *)calloc(outvect_length, sizeof(double));
	//Initialize the weight with a random value between [0,0.05]
	srand((unsigned)time(NULL));
	for (i = 0; i < invect_length*outvect_length; i++)
	{
		fc->weight[i] = 0.05*rand() / RAND_MAX;
	}
}

//Release the memory of the fully-connected layer
void free_fc_layer(FC_LAYER *fc)
{
	FREE(fc->weight);
	FREE(fc->output);
	FREE(fc->sensitivity);
	FREE(fc->gradient);
	FREE(fc->train_label);
	FREE(fc->test_label);

}

//Forward calculation of the convolution layer
void conv_forward(double *input, CONV_LAYER *conv)
{
	int i, j, k, l, vect_len;
	double temp;
	vect_len = (conv->input_length - conv->core_length + 1);
	for (i = 0; i < conv->core_num; i++)
	{
		for (j = 0; j < vect_len; j++)
		{
			temp = conv->bias[i];
			for (k = 0; k < conv->scale; k++)
			{
				for (l = 0; l < conv->core_length; l++)
				{
					temp += input[k*conv->input_length + j + l] *
							conv->weight[i * (conv->core_length * conv->scale)+ 
									     k*conv->core_length + l];
				}
			}
			conv->output[vect_len*i + j] = activation_f(ACTIVATION,temp);
		}
	}
		
}

//Forward calculation of the pooling layer (max pooling)
void max_pooling(double *input, POOLING_LAYER *pooling)
{
	int outvect_length,i,j;
	outvect_length = pooling->invect_length / 2;
	for (i = 0; i < pooling->invect_num; i++)
	{
		for (j = 0; j < outvect_length; j++)
		{
			pooling->output[i*outvect_length + j] =
				MAX(input[i*pooling->invect_length + j * 2],
					input[i*pooling->invect_length + j * 2 + 1]);
			pooling->max_index[i*outvect_length + j] =
				input[i*pooling->invect_length + j * 2] >
				input[i*pooling->invect_length + j * 2 + 1] ?
				(i*pooling->invect_length + j * 2) :
				(i*pooling->invect_length + j * 2 + 1);
		}
	}
}

//Forward calculation of the fully-connected layer
void fc_forward(double *input, FC_LAYER *fc)
{
	int i,j;
	double temp;
	for (i = 0; i < fc->outvect_length; i++)
	{
		temp = fc->bias;
		for (j = 0; j < fc->invect_length; j++)
		{
			temp += input[j] * fc->weight[i*fc->invect_length + j];
		}
		fc->output[i] = activation_f(ACTIVATION,temp);
	}
}

//Back propagation of the fully-connected layer
void fc_back_propagation(double *input, FC_LAYER *fc, int index)
{
	double *vect_label;
	int label_index,i,j;
	vect_label = (double *)malloc(fc->outvect_length*sizeof(double));
	label_index = fc->train_label[index];
	for (i = 0; i < fc->outvect_length; i++)
	{
		if (i == label_index - 1)
			vect_label[i] = 1.0;
		else
			vect_label[i] = 0.0;
	}
	for (i = 0; i < fc->outvect_length; i++)
	{
		fc->sensitivity[i] = (fc->output[i] - vect_label[i])*
							  activation_df(ACTIVATION,fc->output[i]);
		fc->bias_grad += fc->sensitivity[i];
		for (j = 0; j < fc->invect_length; j++)
		{
			fc->gradient[i*fc->invect_length + j] = fc->sensitivity[i] *
													input[j];
		}
	}
	free(vect_label);
}

//update each weight of fully-connected layer
void update_fc_weight(FC_LAYER *fc)
{
	int i, j;
	fc->bias += fc->bias_grad * LEARNING_RATE;
	for (i = 0; i < fc->outvect_length; i++)
	{
		for (j = 0; j < fc->invect_length; j++)
		{
			fc->weight[i*fc->invect_length + j] -=
			fc->gradient[i*fc->invect_length + j] * LEARNING_RATE;
		}
	}
}

//Back propagation of the pooling layer (from fully-connected layer to pooling 
//layer)
void pool_back_propagation1(POOLING_LAYER *pool, FC_LAYER *fc)
{
	int i,j;
	double temp;
	for (i = 0; i < pool->outnode_num; i++)
	{
		temp = 0.0;
		for (j = 0; j < fc->outvect_length; j++)
		{
			temp += fc->sensitivity[j] * fc->weight[j*pool->outnode_num + i];
		}
		pool->sensitivity[i] = temp;
	}
}

//Back propagation of the convolution layer
void conv_back_propagation(double *input, CONV_LAYER *conv, POOLING_LAYER *pool)
{
	int i, j, k, l, vect_len;
	double temp;
	for (i = 0; i < pool->outnode_num; i++)
		conv->sensitivity[pool->max_index[i]] = pool->sensitivity[i];
	vect_len = (conv->input_length - conv->core_length + 1);
	for (i = 0; i < conv->core_num; i++)
	{
		for (j = 0; j < conv->scale; j++)
		{
			for (k = 0; k < conv->core_length; k++)
			{
				temp = 0.0;
				for (l = 0; l < vect_len; l++)
				{
					temp += conv->sensitivity[i*vect_len+l] * input[j*
						    conv->input_length + l + k];
					
				}
				conv->gradient[i*(conv->scale*conv->core_length) +
							   j*conv->core_length + k] = temp;
				
			}
			
		}
	}
	for (i = 0; i < conv->core_num; i++)
		for (l = 0; l < vect_len; l++)
			conv->bias_grad[i] += conv->sensitivity[i*vect_len + l];
}

//update each weight of convolution layer
void update_conv_weight(CONV_LAYER *conv)
{
	int i, j, k, index, vect_len;
	vect_len = (conv->input_length - conv->core_length + 1);
	for (i = 0; i < conv->core_num; i++)
	{
		for (j = 0; j < conv->scale; j++)
		{
			for (k = 0; k < conv->core_length; k++)
			{
				index = i*(conv->scale*conv->core_length) +
					    j*conv->core_length + k;
				conv->weight[index] -=	conv->gradient[index] * LEARNING_RATE;
			}

		}
	}
	for (i = 0; i < conv->core_num; i++)
		conv->bias[i] += conv->bias_grad[i] * LEARNING_RATE;
}

//Back propagation of the pooling layer (from convolution layer to pooling 
//layer)
void pool_back_propagation2(POOLING_LAYER *pool, CONV_LAYER *conv)
{
	int i, j, k, l;
	double temp;
	for (i = 0; i < conv->scale; i++)
	{
		for (j = 0; j < pool->outvect_length; j++)
		{
			temp = 0.0;
			for (l = 0; l < conv->core_num; l++)
			{
				for (k = conv->core_length-1; k >= 0 ; k--)
				{
					if (j - k >= 0 && j - k < conv->output_length)
					{
						temp +=	conv->weight[l*(conv->core_length*conv->scale)
								+ i*conv->core_length +k] * conv->sensitivity[l*
								conv->output_length + j - k];
					}
				
				}
			}
			pool->sensitivity[i*pool->outvect_length + j] = temp*activation_df(
				ACTIVATION,pool->output[i*pool->outvect_length + j]);
		}
	}
}

//calculate the output error of network, which is the sum of squares of errors 
//for all output layer nodes
double network_error(double *actual, double *expect, int length)
{
	int i;
	double temp = 0.0;
	for (i = 0; i < length; i++)
	{
		temp += 0.5*(expect[i] - actual[i])*(expect[i] - actual[i]);
	}
	return temp;
}

//check the gradient of fully-connected layer
void fc_check_grad(double test_array[], FC_LAYER *fc, int index)
{
	double *vect_label, epsilon = 0.00001, error1, error2, expect_grad,temp;
	int label_index, i;
	vect_label = (double *)malloc(fc->outvect_length * sizeof(double));
	label_index = fc->train_label[index];
	for (i = 0; i < fc->outvect_length; i++)
	{
		if (i == label_index - 1)
			vect_label[i] = 1.0;
		else
			vect_label[i] = 0.0;
	}
	printf("fc_gradient check result:\n");
	for (i = 0; i < fc->outvect_length*fc->invect_length; i++)
	{
		temp = fc->weight[i];
		//Add a small value to the weight
		fc->weight[i] += epsilon;
		fc_forward(test_array, fc);
		error1 = network_error(fc->output, vect_label, fc->outvect_length);

		//Subtract double of the small value from the weight
		fc->weight[i] -= epsilon * 2;
		fc_forward(test_array, fc);
		error2 = network_error(fc->output, vect_label, fc->outvect_length);
		expect_grad = (error1 - error2) / (2 * epsilon);

		//Calculate the gradient and make the comparison
		printf("actual_grad:expect_grad = %.5f:%.5f\n", fc->gradient[i], 
			   expect_grad);
		fc->weight[i] = temp;
	}
	temp = fc->bias;
	//Add a small value to the bias
	fc->bias += epsilon;
	fc_forward(test_array, fc);
	error1 = network_error(fc->output, vect_label, fc->outvect_length);

	//Subtract double of the small value from the bias
	fc->bias -= epsilon * 2;
	fc_forward(test_array, fc);
	error2 = network_error(fc->output, vect_label, fc->outvect_length);

	//Calculate the gradient and make the comparison
	expect_grad = (error1 - error2) / (2 * epsilon);
	printf("bias_grad_actual:bias_grad_expect = %.5f:%.5f\n", fc->bias_grad,
		expect_grad);
	fc->bias = temp;
	free(vect_label);
}

//calculate the output error of convolution layer, which is the sum of output 
//of layer nodes
double conv_error(CONV_LAYER *conv)
{
	int i;
	double temp = 0.0;
	for (i = 0; i < conv->output_node_num; i++)
	{
		temp += conv->output[i];
	}
	return temp;
}


//check the gradient of one convolution layer, using identity function as 
//activation function
void conv_check_grad2(double test_array[], CONV_LAYER *conv)
{
	double epsilon = 0.00001, error1, error2, expect_grad, temp;
	int i;
	printf("conv_gradient check result:\n");
	for (i = 0; i < conv->core_num*conv->core_length*conv->scale; i++)
	{
		temp = conv->weight[i];
		conv->weight[i] += epsilon;
		conv_forward(test_array, conv);
		error1 = conv_error(conv);
		conv->weight[i] -= epsilon * 2;
		conv_forward(test_array, conv);
		error2 = conv_error(conv);
		expect_grad = (error1 - error2) / (2 * epsilon);
		printf("actual_grad:expect_grad = %.5f:%.5f\n", conv->gradient[i],
			   expect_grad);
		conv->weight[i] = temp;
	}
	
}


//check the gradient of convolution layer in a LeNet-5 like architecture network
void conv_check_grad1(double test_array[], CONV_LAYER *conv1, POOLING_LAYER 
					  *pooling1, CONV_LAYER *conv2, POOLING_LAYER *pooling2, 
					  FC_LAYER *fc, int index)
{
	double *vect_label, epsilon = 0.00001, error1, error2, expect_grad, temp;
	int label_index, i;
	vect_label = (double *)malloc(fc->outvect_length * sizeof(double));
	label_index = fc->train_label[index];
	for (i = 0; i < fc->outvect_length; i++)
	{
		if (i == label_index - 1)
			vect_label[i] = 1.0;
		else
			vect_label[i] = 0.0;
	}
	printf("conv_gradient check result:\n");
	for (i = 0; i < conv1->core_num*conv1->core_length*conv1->scale; i++)
	{
		temp = conv1->weight[i];
		//Add a small value to the weight
		conv1->weight[i] += epsilon;
		conv_forward(test_array, conv1);
		max_pooling(conv1->output, pooling1);
		conv_forward(pooling1->output, conv2);
		max_pooling(conv2->output, pooling2);
		fc_forward(pooling2->output, fc);
		error1 = network_error(fc->output, vect_label, fc->outvect_length);

		//Subtract double of the small value from the weight
		conv1->weight[i] -= epsilon * 2;
		conv_forward(test_array, conv1);
		max_pooling(conv1->output, pooling1);
		conv_forward(pooling1->output, conv2);
		max_pooling(conv2->output, pooling2);
		fc_forward(pooling2->output, fc);
		error2 = network_error(fc->output, vect_label, fc->outvect_length);

		//Calculate the gradient and make the comparison
		expect_grad = (error1 - error2) / (2 * epsilon);
		printf("actual_grad:expect_grad = %.5f:%.5f\n", conv1->gradient[i],
			expect_grad);
		conv1->weight[i] = temp;
	}
	for (i = 0; i < conv1->core_num; i++)
	{
		temp = conv1->bias[i];
		//Add a small value to the bias
		conv1->bias[i] += epsilon;
		conv_forward(test_array, conv1);
		max_pooling(conv1->output, pooling1);
		conv_forward(pooling1->output, conv2);
		max_pooling(conv2->output, pooling2);
		fc_forward(pooling2->output, fc);
		error1 = network_error(fc->output, vect_label, fc->outvect_length);

		//Subtract double of the small value from the bias
		conv1->bias[i] -= epsilon * 2;
		conv_forward(test_array, conv1);
		max_pooling(conv1->output, pooling1);
		conv_forward(pooling1->output, conv2);
		max_pooling(conv2->output, pooling2);
		fc_forward(pooling2->output, fc);
		error2 = network_error(fc->output, vect_label, fc->outvect_length);

		//Calculate the gradient and make the comparison
		expect_grad = (error1 - error2) / (2 * epsilon);
		printf("bias grad check: %d\nactual_grad:expect_grad = %.5f:%.5f\n", 
			   i, conv1->bias_grad[i], expect_grad);
		conv1->bias[i] = temp;
	}
	free(vect_label);
}


#if defined(NETWORK_GRAD_CHECK) || defined(CONV_LAYER_GRAD_CHECK) || \
								   defined(FC_LAYER_GRAD_CHECK)
int main()
{
	double test_array[30] = {2,1,0,2,1,1,2,0,0,0,0,0,1,1,1,
							 1,4,1,1,1,1,2,0,0,0,2,2,1,1,3};
	FC_LAYER fc;
	CONV_LAYER conv1,conv2;
	POOLING_LAYER pooling1,pooling2;
	char *train_data, *train_label, *test_data, *test_label;
#ifdef CONV_LAYER_GRAD_CHECK
	int i;
#endif

	train_data = "../train_sample.tcnn";
	train_label = "../train_label.tcnn";
	test_data = "../test_sample.tcnn";
	test_label = "../test_label.tcnn";

//check the gradient of convolution layer in a LeNet-5 like architecture network	
#ifdef NETWORK_GRAD_CHECK
	initial_conv_layer(&conv1, 30, 5, 1, 3);
	initial_pooling_layer(&pooling1, 26, 3);
	initial_conv_layer(&conv2, 13, 4, 3, 3);
	initial_pooling_layer(&pooling2, 10, 3);
	initial_fc_layer(&fc, 15, 6, train_label, test_label);
	conv_forward(test_array, &conv1);
	max_pooling(conv1.output, &pooling1);
	conv_forward(pooling1.output, &conv2);
	max_pooling(conv2.output, &pooling2);
	fc_forward(pooling2.output, &fc);
	fc_back_propagation(pooling2.output, &fc, 0);
	pool_back_propagation1(&pooling2, &fc);
	conv_back_propagation(pooling1.output, &conv2, &pooling2);
	pool_back_propagation2(&pooling1, &conv2);
	conv_back_propagation(test_array, &conv1, &pooling1);
	conv_check_grad1(test_array, &conv1, &pooling1, &conv2, &pooling2, &fc, 0);
	//Release the memory occupied by the pointer in each layer
	free_conv_layer(&conv1);
	free_pooling_layer(&pooling1);
	free_conv_layer(&conv2);
	free_pooling_layer(&pooling2);
	free_fc_layer(&fc);
#endif

//check the gradient of one convolution layer, using identity function as 
//activation function
#ifdef CONV_LAYER_GRAD_CHECK
	initial_conv_layer(&conv1, 30, 5, 1, 3);
	initial_pooling_layer(&pooling1, 26, 3);
	for (i = 0; i < conv1.output_node_num; i++)
		conv1.sensitivity[i] = 1.0;
	conv_forward(test_array, &conv1);
	conv_back_propagation(test_array, &conv1, &pooling1);
	conv_check_grad2(test_array, &conv1);
	//Release the memory occupied by the pointer in each layer
	free_conv_layer(&conv1);
	free_pooling_layer(&pooling1);
#endif

//check the gradient of fully-connected layer
#ifdef FC_LAYER_GRAD_CHECK
	initial_fc_layer(&fc, 30, 6, train_label, test_label);
	fc_forward(test_array, &fc);
	fc_back_propagation(test_array, &fc, 0);
	fc_check_grad(test_array, &fc, 0);
	//Release the memory occupied by the pointer in each layer
	free_fc_layer(&fc);
#endif
	
#ifdef _WIN32
	system("pause");
#endif
	return 0;

}
#endif
