# wave_cnn
wave_cnn is a C implementation of convolutional neural network (CNN). It is 
specifically designed to handle one-dimensional signals and easy to understand. 
therefore, it is a good reference material to walk into the world of deep neural
networks.

## Features
- reasonably fast, since it is designed to handle one-dimensional signals
	- old computer can run wave_cnn quickly
	- training of over 4000 samples only costs 5 seconds in a LeNet-5 like 
	architecture (Intel i5-6400)
- dependency-free & portable
	- Run anywhere as long as you have a compiler which supports C
	- you only need to write your model in C and include "global.h" & 
	"layer_function.h"
- easy-understanding
	- core function are writen within a page and there is no complex stucture
	- it is designed to handle one-dimensional signals so that no mattix 
	calculation was included
	
## Supported networks
- layer types
	- fully-connected layer
	- convolutional layer
	- max pooling layer
- activation functions
	- tanh
	- ReLu
	- leaky-ReLu
	
## Examples
construct convolutional neural networks
```
#include <stdio.h>
#include "global.h"
#include "layer_function.h"

//local functions
void train(INPUT_LAYER *input, CONV_LAYER *conv1, POOLING_LAYER *pooling1, 
		   CONV_LAYER *conv2, POOLING_LAYER *pooling2, FC_LAYER *output);
double check_accuracy(INPUT_LAYER *input, CONV_LAYER *conv1, 
					  POOLING_LAYER *pooling1,CONV_LAYER *conv2, 
					  POOLING_LAYER *pooling2, FC_LAYER *output);

//The example runs only if the following three macro definitions are commented 
//out
#if !defined(NETWORK_GRAD_CHECK) && !defined(CONV_LAYER_GRAD_CHECK) && \
                                    !defined(FC_LAYER_GRAD_CHECK)
//======Example:convolutional neural networks (LeNet-5 like architecture)======
int main()
{
	char *train_data, *train_label, *test_data, *test_label;
	INPUT_LAYER input;
	CONV_LAYER conv1, conv2;
	POOLING_LAYER pooling1, pooling2;
	FC_LAYER output;

	//Enter the file path of the test dataset and training dataset
	train_data = "../data/train_sample.tcnn";
	train_label = "../data/train_label.tcnn";
	test_data = "../data/test_sample.tcnn";
	test_label = "../data/test_label.tcnn";

	//Initialize each layer
	initial_input_layer(&input, train_data, test_data);
	initial_conv_layer(&conv1, INPUT_LENGTH, 15, 1, 3);
	initial_pooling_layer(&pooling1, 102, 3);
	initial_conv_layer(&conv2, 51, 22, 3, 3);
	initial_pooling_layer(&pooling2, 30, 3);
	initial_fc_layer(&output, 15*3, 6, train_label,test_label);

	//Start training the neural network
	train(&input, &conv1, &pooling1, &conv2, &pooling2, &output);
	printf("training complete!\n");

	//Release the memory occupied by the pointer in each layer
	free_input_layer(&input);
	free_conv_layer(&conv1);
	free_pooling_layer(&pooling1);
	free_conv_layer(&conv2);
	free_pooling_layer(&pooling2);
	free_fc_layer(&output);
#ifdef _WIN32
	system("pause");
#endif
	return 0;
}
#endif

//The following train function should be adjusted if the structure was changed 
void train(INPUT_LAYER *input, CONV_LAYER *conv1, POOLING_LAYER *pooling1,
	       CONV_LAYER *conv2, POOLING_LAYER *pooling2, FC_LAYER *output)
{
	int i,j;
	double accuracy1 = 0.0, accuracy2 = 0.0;
	
	for (i = 0; i < input->train_sample_num/BATCH_SIZE; i++)
	{
		for (j = 0; j < BATCH_SIZE; j++)
		{
			//Forward calculation layer by layer
			conv_forward(input->train_data+(i*BATCH_SIZE+j)*INPUT_LENGTH,conv1);
			max_pooling(conv1->output, pooling1);
			conv_forward(pooling1->output, conv2);
			max_pooling(conv2->output, pooling2);
			fc_forward(pooling2->output, output);

			//Back propagation layer by layer
			fc_back_propagation(pooling2->output, output, (i*BATCH_SIZE + j));
			pool_back_propagation1(pooling2, output);
			conv_back_propagation(pooling1->output, conv2, pooling2);
			pool_back_propagation2(pooling1, conv2);
			conv_back_propagation(input->train_data + (i*BATCH_SIZE + j)*
                                  INPUT_LENGTH, conv1, pooling1);

			//update each weight
			update_fc_weight(output);
			update_conv_weight(conv2);
			update_conv_weight(conv1);
		}
		printf("training times : %d\n", i + 1);

		//check the accuracy after each batch training was finished, If the 
		//accuracy rate drops, stop training  
		accuracy2 = check_accuracy(input,conv1,pooling1,conv2,pooling2,output);
		printf("accuracy = %.5f\n", accuracy2);
		if (accuracy2 < accuracy1)
			break;
		else
			accuracy1 = accuracy2;
	}
	

}


//get the index of the max value in output array
int get_max(FC_LAYER *fc)
{
	int i, index = 0;
	double temp;
	temp = fc->output[0];
	for (i = 1; i < fc->outvect_length; i++)
	{
		if (fc->output[i] > temp)
		{
			index = i;
			temp = fc->output[i];
		}
			
	}
	return index;
}

//Return the accuracy after each training for a batch 
double check_accuracy(INPUT_LAYER *input, CONV_LAYER *conv1,
					  POOLING_LAYER *pooling1, CONV_LAYER *conv2,
					  POOLING_LAYER *pooling2, FC_LAYER *output)
{
	int correct_num = 0,i,result;
	for (i = 0; i < input->test_sample_num; i++)
	{
		//Forward calculation layer by layer
		conv_forward(input->test_data + i*INPUT_LENGTH, conv1);
		max_pooling(conv1->output, pooling1);
		conv_forward(pooling1->output, conv2);
		max_pooling(conv2->output, pooling2);
		fc_forward(pooling2->output, output);
		result = get_max(output)+1;

		//compare output of NN with label
		if (result == output->test_label[i])
			correct_num++;
	}
	printf("test set accuracy:\ncorrect_num/total_num = %d/%d\n", correct_num,
		   input->test_sample_num);
	return (double)correct_num / input->test_sample_num;
}
```
	