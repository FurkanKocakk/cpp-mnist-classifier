#pragma once
#include <cmath>
#include "pch.h"

float* Add_Data(float* sample, int Size, float* x, int Dim);
float* Add_Labels(float* Labels, int Size, int label);
float* init_array_random(int len);
void Z_Score_Parameters(float* x, int Size, int dim, float* mean, float* std);
float sgn_net(float net);
int Test_Forward(float* x, float* weight, float* bias, int neuron_count, int inputDim);
float* Z_Score_Norm(float* Samples, int numSample, int inputDim);
/*regression(+LineCiz) ve train_fcn'yi network.h'a ekle, grafiði formun sað altýna al, 
training hatalý, picture box ve mouse sürükleme ekle*/