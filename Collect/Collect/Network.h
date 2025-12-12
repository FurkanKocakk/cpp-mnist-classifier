#pragma once
#include "Process.h"
#include "pch.h"

// topology: Her katmandaki nöron sayısı (örn: {inputDim, hidden1, hidden2, ...,
// outputDim}) layer_count: Toplam katman sayısı (input dahil) Weights: Her
// katman arası ağırlık matrisleri dizisi (Weights[0]: Input->Hidden1,
// Weights[1]: Hidden1->Hidden2...) Biases: Her katman için bias dizisi
float *train_fcn(float *Samples, int numSample, float *targets, int *topology,
                 int layer_count, float **Weights, float **Biases,
                 float learning_rate, float momentum, float Min_Err,
                 int Max_epoch, int &epoch);
// Output: Class Index (0 to class_count-1)
int Test_Forward(float *input, float **Weights, float **Biases, int *topology,
                 int layer_count);

// MLP Regression (1 Input -> Hidden... -> 1 Output)
// Returns error history array
float *train_mlp_regression(float *Samples, int numSample, float *targets,
                            int *topology, int layer_count, float **Weights,
                            float **Biases, float learning_rate, float momentum,
                            float Min_Err, int Max_epoch, int &epoch);

float Evaluate_Regression_Point(float input_val, float **Weights,
                                float **Biases, int *topology, int layer_count);
