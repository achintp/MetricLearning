#include "itml.h"
#include <numeric>
#include <cmath>
#include <iostream>
#include <stdlib.h>

long double norm(vector<float> features){
	return 0;
}

void ITML_Alg::_check_constraints(int C, int N, int M){
	bool remake = false;
	float thresh = 1e-10;
	// Check whether the points are the same
	for(int i = 0; i < C; i++){
		int i1 = constraints[i*4];
		int i2 = constraints[1 + i*4];
		vector<float> features;
		for(int j = 0; j < M; j++){
			features.push_back(data[i1*M +j] - data[i2*M + j]);
		}
		if(norm(features) < thresh){
			cout << "Constraints between same points";
			exit(EXIT_FAILURE);				
		}
	}
}

float* ITML(){
	int i = 1;
	int iter = 0;

	float* lambda = new float[C];
	memset(lambda, 0, sizeof(float)*C);
	float* lambda_old = new float[C];
	memset(lambda, 0, sizeof(float)*C);

	float* bounds = new float[C];
	for(int i = 0; i < C; i++){
		bounds[i] = constraints[4*i + 3];
	}

	float* A = new float[N*M];
	for(int i = 0; i < N; i++){
		for(int j = 0; j < M; j++){
			A[i*M + j] = A0[i*M + j];
		}
	}

	float conv = 1e10;

	while(1){
		int i1 = C[i*4];
		int i2 = C[i*4 + 1];
		vector<float> features;
		for(int i = 0; i < M; i++){
			features.push_back(data[i1*M + i] - data[i2*M + i]);
		}
		
	}
}
