#include "mex.h"
#include "class_handle.hpp"
#include "../random_forest.h"
#include <string>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// Get the command string
	char cmd[64];
	if(nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be command string of < 64 chars");

	// Make new object
	if(!strcmp("new", cmd)){
		if (nrhs < 9)
			mexErrMsgTxt("Not enough params to construct forest");

		// Get the forest constructor arguments
		// data should be atts x insts
		// labels should be insts x 1
		float *inp = (float *)mxGetData(prhs[1]);
		int insts = mxGetM(prhs[1]); // Number of rows in a matrix
		int atts = mxGetN(prhs[1]); // Number of columns in a matrix
		float *data = new float[atts*insts];
		int *labels = (int *)mxGetData(prhs[2]);
		int insts2 = mxGetM(prhs[2]); // Number of rows
		int num_trees = mxGetScalar(prhs[3]);
		int K = mxGetScalar(prhs[4]);
		int F = mxGetScalar(prhs[5]);
		int min_size = mxGetScalar(prhs[6]);
		int splitfun = mxGetScalar(prhs[7]);
		int threads = mxGetScalar(prhs[8]);
		bool compute_unique = false;
		int counter = 0;

		cout << atts << insts << endl;
		//cout << data[0] << " " << data[1] << endl;
		for(int i = 0; i < atts; i++){
			for(int j = 0; j < insts; j++){
				data[i*insts + j] = inp[counter++];
				// cout << data[i*insts + j] << " ";
			}
			cout << "Next" << endl;
		}

		if(nlhs < 1)
			mexErrMsgTxt("New: Expected to return an object");
		try{
			plhs[0] = convertPtr2Mat<RandomForest>(new RandomForest(data, atts, insts, labels, 
			insts2, num_trees, K, F, min_size, splitfun, threads, compute_unique));
		}catch(...){
			mexErrMsgTxt("Failed");
		}
		delete[] data;
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

	RandomForest *rfd = convertMat2Ptr<RandomForest>(prhs[1]);

	if(!strcmp("getCodes", cmd)){
		if(nrhs < 3)
			mexErrMsgTxt("Too less arguments for getting codes");
		if(nlhs == 0)
			mexErrMsgTxt("Needs to output codes");

		float *inp = (float *)mxGetData(prhs[2]);
		int atts = mxGetM(prhs[2]);  // Get cols
		int insts =  mxGetN(prhs[2]);  // Get rows
		float *data = new float[insts*atts];
		int trsize = mxGetScalar(prhs[3]);
		double *codes = new double[insts*trsize];
		bool position = false;
		int counter = 0;

		for(int i = 0; i < insts; i++){
			for(int j = 0; j < atts; j++){
				data[i*atts + j] = inp[counter++];
				// cout << data[i*atts + j] << " ";
			}
			cout << endl;
		}

		rfd->rfdcode(data, insts, atts, codes, insts, trsize, position);

		plhs[0] = mxCreateDoubleMatrix(insts, trsize, mxREAL);
		double *out = (double *)mxGetData(plhs[0]);
		for(int i = 0; i < insts; i++){
			for(int j = 0; j < trsize; j++){
				out[i*trsize + j] = codes[i*trsize + j];
			}
		}

		delete[] codes;
		delete[] data;
		return;
	}

	// Get the class instance pointer from the arguments
	RandomForest *rfd_instance = convertMat2Ptr<RandomForest>(prhs[1]);

	mexErrMsgTxt("Command not recognized");
}
