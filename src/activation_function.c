#include <math.h>
#include "global.h"
#include "activation_function.h"

double relu_f(double a)
{
	return MAX(a, (double)0);
}

double relu_df(double a)
{
	return a > (double)0 ? (double)1 : (double)0;
}

double leaky_relu_f(double a)
{
	return a > (double)0 ? a : a * (double)0.01;
}

double leaky_relu_df(double a)
{
	return a > (double)0 ? (double)1 : (double)0.01;
}

double tanH_df(double y)
{
	return 1 - y * y;
}

//Activation Function
double activation_f(activation f, double a)
{
	switch (f)
	{
	case relu:
		return relu_f(a);
	case leaky_relu:
		return leaky_relu_f(a);
	case tanH:
		return tanh(a);
	default:
		return 0.0;
	}
}

//Derivative of activation function, input y refers to the output value of 
//activation function
double activation_df(activation f, double y)
{
	switch (f)
	{
	case relu:
		return relu_df(y);
	case leaky_relu:
		return leaky_relu_df(y);
	case tanH:
		return tanH_df(y);
	default:
		return 0.0;
	}
}