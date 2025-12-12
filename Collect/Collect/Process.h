#pragma once
#include "pch.h"
#include <cmath>

float *Add_Data(float *sample, int Size, float *x, int Dim);
float *Add_Labels(float *Labels, int Size, int label);
float *init_array_random(int len);
void Z_Score_Parameters(float *x, int Size, int dim, float *mean, float *std);
float sgn_net(float net);
int Test_Forward(float *x, float **Weights, float **Biases, int *topology,
                 int layer_count);
float *Z_Score_Norm(float *Samples, int numSample, int inputDim);
