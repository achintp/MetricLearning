#include "mex.h"
#include "class_handle.hpp"
#include "random_forest.h"

void mexFUnction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// Get the command string
	char cmd[64];
	if(nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be command string of < 64 chars");

	// Make new object
	if(!strcmp("new", cmd)){
		if (nrhs < 13)
			mexErrMsgTxt("Not enough params to construct forest");

		// Get the forest constructor arguments
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

		if(nlhs < 1)
			mexErrMsgTxt("New: Expected to return an object");
		plhs[0] = convertPtr2Mat<RandomForest>(new RandomForest(data, N, M, labels, 
		insts2, num_trees, K, F, min_size, splitfun, threads, compute_unique));
		return;
	}

	// check if there is a second input given
	if(nrhs < 2)
		mexErrMsgTxt("Second input should be the class instance handle");

	// delete the object
	if(!strcmp("delete", cmd)){
		destroyObject<RandomForest>(prhs[1]);
		if(nlhs != 0 || nrhs != 2)
			mexWarnMsgTxt("Delete: something was ignored");
		return;
	}

	// Get the class instance pointer from the arguments
	RandomForest *rfd_instance = convertMat2Ptr<RandomForest>(prhs[1]);

	mexErrMsgTxt("Command not recognized");
}
