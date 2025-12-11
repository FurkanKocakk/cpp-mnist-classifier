#include "pch.h"
#include "Network.h"
#include "Process.h"

float* train_fcn(float* Samples, int numSample, float* targets, int inputDim, int class_count, float* Weights, float* bias, float learning_rate, float Min_Err, int Max_epoch, int& epoch) {

	float total_err;
	float* temp = new float[Max_epoch];
	float* net, * fnet, * f_der, * desired, * err, * delta;
	net = new float[class_count];
	fnet = new float[class_count];
	f_der = new float[class_count];
	desired = new float[class_count];
	err = new float[class_count];
	delta = new float[class_count];
	epoch = 0;

	do {
		total_err = 0.0f;
		for (int step = 0; step < numSample; step++) {
			//FeedForward
			for (int k = 0; k < class_count;k++) {
				net[k] = 0.0f;
				for (int i = 0; i < inputDim; i++) {
					net[k] += Weights[k * inputDim + i] * Samples[step * inputDim + i];
				}
					net[k] += bias[k];
					fnet[k] = (2.0f / (float)(1.0f + exp(-net[k])) - 1.0f);
					f_der[k] = 0.5f * (1.0f - fnet[k] * fnet[k]);
				
				for (int k = 0; k < class_count; k++) {
					if (targets[step] == k) {
						desired[k] = 1.0;
					}
					else desired[k] = -1.0;
					//Backward
					err[k] = desired[k] - fnet[k];
					delta[k] = learning_rate * err[k] * f_der[k];
					for (int i = 0; i < inputDim; i++) {
						Weights[k * inputDim + i] += delta[k] * Samples[step * inputDim + i];
						bias[k] += delta[k];
						total_err += (0.5f * (err[k] * err[k]));
					}
				}// for(step)
				total_err /= float(class_count * numSample);
				if (epoch < Max_epoch) {
					temp[epoch] = total_err;
					epoch++;
				}
			}
		}
	} while ((total_err > Min_Err) && (epoch < Max_epoch));
	delete[] net;
	delete[] fnet;
	delete[] f_der;
	delete[] desired;
	delete[] err;
	delete[] delta;
	return temp;
} //train

