/*
  Copyright (c) 2017, Jiemeng Zou
  All rights reserved.
  Use of this source code is governed by a BSD-style license that can be found
  in the LICENSE file.
*/
#ifndef TWAVE_CNN_LAYER_FUNCTION_H
#define TWAVE_CNN_LAYER_FUNCTION_H

#include "global.h"
#include "layer.h"

void initial_input_layer(INPUT_LAYER *input, const char *train_file,
						 const char *test_file);
void initial_conv_layer(CONV_LAYER *conv, int input_length, int core_length,
						int scale, int core_num);
void initial_pooling_layer(POOLING_LAYER *pooling, int invect_length,
						   int invect_num);
void initial_fc_layer(FC_LAYER *fc, int invect_length, int outvect_length, 
					  const char *train_file, const char *test_file);
void conv_forward(double *input, CONV_LAYER *conv);
void max_pooling(double *input, POOLING_LAYER *pooling);
void fc_forward(double *input, FC_LAYER *fc);
void fc_back_propagation(double *input, FC_LAYER *fc, int index);
void update_fc_weight(FC_LAYER *fc);
void pool_back_propagation1(POOLING_LAYER *pool, FC_LAYER *fc);
void conv_back_propagation(double *input, CONV_LAYER *conv,
						   POOLING_LAYER *pool);
void update_conv_weight(CONV_LAYER *conv);
void pool_back_propagation2(POOLING_LAYER *pool, CONV_LAYER *conv);
void free_input_layer(INPUT_LAYER *input);
void free_conv_layer(CONV_LAYER *conv);
void free_pooling_layer(POOLING_LAYER *pooling);
void free_fc_layer(FC_LAYER *fc);

#endif
