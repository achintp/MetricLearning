#include "mex.h"
#include "random_forest.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

	// Get the forest constructor arguments
	if(nrhs < 10)
		mexErrMsgTxt("Not enough inputs");

	float *data = (float *)mxGetData(prhs[1]);
	int N = mxGetM(prhs[1]);
	int M = mxGetN(prhs[1]);
	int *labels = (int *)mxGetData(prhs[2]);
	int insts2 = mxGetN(prhs[2]);
	int num_trees = mxGetScalar(prhs[3]);
	int K = mxGetScalar(prhs[4]);
	int F = mxGetScalar(prhs[5]);
	int min_size = mxGetScalar(prhs[6]);
	int splitfun = mxGetScalar(prhs[7]);
	int threads = mxGetScalar(prhs[8]);
	bool compute_unique = false;

	RandomForest *rfd = new RandomForest(data, N, M, labels, 
		insts2, num_trees, K, F, min_size, splitfun, threads, compute_unique);

	float *test_data = (float *)mxGetData(prhs[9]);
	int atts = mxGetN(prhs[9]);
	int insts =  mxGetM(prhs[9]);
	int trsize = num_trees;
	float *codes = new float[insts*trsize];
	bool position = false;

	rfd->rfdcode(test_data, insts, atts, codes, insts, trsize, position);
	plhs[0] = mxCreateNumericMatrix(insts, trsize, mxSINGLE_CLASS, mxREAL);
		float *out = (float *)mxGetData(plhs[0]);
		for(int i = 0; i < insts; i++){
			for(int j = 0; j < trsize; j++){
				out[i*trsize + j] = codes[i*trsize + j];
			}
		}

	delete[] codes;
	return;
}