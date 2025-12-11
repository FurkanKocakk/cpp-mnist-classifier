#pragma once
#include "pch.h"
#include "Process.h"

float* train_fcn(float* Samples, int numSample, float* targets, int inputDim, int class_count, float* Weights, float* bias, float learning_rate, float Min_Err, int Max_epoch, int& epoch);