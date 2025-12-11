#include "pch.h"
#include "Process.h"
#include <iostream>

float* Add_Data(float* sample, int Size, float* x, int Dim) {
	float* temp = new float[Size * Dim];
	if (sample) {
		for (int i = 0; i < (Size - 1) * Dim; i++)
			temp[i] = sample[i];
		delete[] sample;
	}
	for (int i = 0; i < Dim; i++)
		temp[(Size - 1) * Dim + i] = x[i];
	return temp;
}
float* Add_Labels(float* Labels, int Size, int label) {
	float* temp = new float[Size];
	if (Labels) {
		for (int i = 0; i < Size - 1; i++)
			temp[i] = Labels[i];
		delete[] Labels;
	}
	temp[Size - 1] = float(label);
	return temp;
}
float* init_array_random(int len) {
	float* arr = new float[len];
	for (int i = 0; i < len; i++)
		arr[i] = ((float)rand() / RAND_MAX) - 0.5f;
	return arr;
}
void Z_Score_Parameters(float* x, int Size, int dim, float* mean, float* std) {

	float* Total = new float[dim];

	int i, j;
	for (i = 0; i < dim; i++) {
		mean[i] = std[i] = Total[i] = 0.0;
	}
	for (i = 0; i < Size; i++)
		for (j = 0; j < dim; j++)
			Total[j] += x[i * dim + j];
	for (i = 0; i < dim; i++)
		mean[i] = Total[i] / float(Size);

	for (i = 0; i < Size; i++)
		for (j = 0; j < dim; j++)
			std[j] += ((x[i * dim + j] - mean[j]) * (x[i * dim + j] - mean[j]));

	for (j = 0; j < dim; j++)
		std[j] = sqrt(std[j] / float(Size));

	delete[] Total;

}//Z_Score_Parameters
int Test_Forward(float* x, float* weight, float* bias, int num_Class, int inputDim) {
	int i, j, index_Max;
	if (num_Class > 2) {
		float* output = new float[num_Class];
		// Calculation of the output layer input
		for (i = 0; i < num_Class; i++) {
			output[i] = 0.0f;
			for (j = 0; j < inputDim; j++)
				output[i] += weight[i * inputDim + j] * x[j];
			output[i] += bias[i];
		}
		for (i = 0; i < num_Class; i++)
			output[i] = tanh(output[i]);

		//Find Maximum in neuron
		float temp = output[0];
		index_Max = 0;
		for (i = 1; i < num_Class; i++)
			if (temp < output[i]) {
				temp = output[i];
				index_Max = i;
			}

		delete[] output;
	}
	else {
		float output = 0.0f;
		for (j = 0; j < inputDim; j++)
			output += weight[j] * x[j];
		output += bias[0];
		output = tanh(output);
		if (output > 0.0f)
			index_Max = 0;
		else index_Max = 1;
	}
	return index_Max;

}//
float sgn_net(float net){
	if (net >= 0)
		return 1.0;
	else
		return -1.0;
};
float* Z_Score_Norm(float* Samples, int numSample, int inputDim) {
	float* normSamples = new float[numSample * inputDim];
	float* mean = new float[inputDim];
	float* std = new float[inputDim];

	// 1. Ortalamalarý hesapla
	for (int j = 0; j < inputDim; j++) {
		mean[j] = 0.0f;
		for (int i = 0; i < numSample; i++) {
			mean[j] += Samples[i * inputDim + j];
		}
		mean[j] /= numSample;
	}

	// 2. Standart sapmalarý hesapla
	for (int j = 0; j < inputDim; j++) {
		std[j] = 0.0f;
		for (int i = 0; i < numSample; i++) {
			float diff = Samples[i * inputDim + j] - mean[j];
			std[j] += diff * diff;
		}
		std[j] = sqrt(std[j] / numSample);
		if (std[j] == 0) std[j] = 1.0f; // Bölme sýfýr olmasýn
	}

	// 3. Normalize et
	for (int i = 0; i < numSample; i++) {
		for (int j = 0; j < inputDim; j++) {
			normSamples[i * inputDim + j] = (Samples[i * inputDim + j] - mean[j]) / std[j];
		}
	}

	// Bellek temizliði
	delete[] mean;
	delete[] std;

	return normSamples;
}
