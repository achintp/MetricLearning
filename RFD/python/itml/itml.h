/**
ITML algorithm
**/

#ifndef _ITML_H
#define _ITML_H
#include <iostream>
#include <vector>
#include <assert.h>
#include <unordered_set>

using namespace std;

class ITML_Alg{
public:
	float gamma;
	float tolerance;
	float max_iters;
	float* constraints;		//Cx4 matrix
	float* A0;		// MxM matrix							
	float* data;		// NxM matrix 
	int C;
	int M;
	int N;

	ITML_Alg(vector<float> params, float* cons, float* data, float* reg, int num_cons, int num_features, int num_samples):
		gamma(params[0]), tolerance(params[1]), max_iters(params[2]), constraints(cons),
		A0(reg), data(data), C(num_cons), M(num_features), N(num_features){}
	~ITML_Alg(){}


private:
	void _check_constraints(int C, int M, int N);
};

#endif