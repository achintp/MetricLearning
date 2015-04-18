//Swig interface file for librf code

%module librf

%{
#define SWIG_FILE_WITH_INIT
#include "librf/random_forest.h"
%}

%feature("autodoc", "1");

%include "numpy.i"

%init 
%{
import_array();
%}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {	(const float* data, int atts, int insts),
					    						(const float* data, int insts, int atts),
					    						(const float* data2, int insts2, int atts2),
											    (const float* points, int m1, int atts),
											    (const float* p1, int m1, int d1),
											    (const float* p2, int m2, int d2),
											    (const float* p1, const int n, const int d1),
												(const float* p2, const int m, const int d2)};
%apply (int* IN_ARRAY1, int DIM1) {(const int *labels, int insts2)};
%apply (int* IN_ARRAY2, int DIM1, int DIM2) {(const int* constraints, int num_cons, int three)};
%apply (signed char* INPLACE_ARRAY2, int DIM1, int DIM2) {(signed char* votes, int insts3, int ntrees)};
%apply (signed char* INPLACE_ARRAY1, int DIM1) {(signed char* preds, int insts4)};
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* vars, int atts),
                                          (float* codes, int insts1)}
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* dists, int insts12, int insts22),
													(float* regs, int insts2, int C),
													(float* dists, int m3, int nn2),
													(float* out, const int n2, const int m2)
                                                    (float* codes, int d1, int d2),
                                                    (float* codes, int insts1, int trsize)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* neighbors, int m2, int nn1),
												  (int* pairs, int numpairs, int three)}
%apply (float* ARGOUT_ARRAY1, int DIM1) {(float* out, int m3)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *codes, int insts1, int trsize)}

%rename (votesset) vs;
%exception vs {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}

%rename (nearest) nn;
%exception nn {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}

%rename (predict) pr;
%exception pr {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}

%rename (testing_accuracy) ta;
%exception ta {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}

%rename(RandomForest_cons) RandomForest(const float* data, int atts, int insts, const int* constraints, int num_cons, int three, int num_trees,
			int K, int F, int min_size, int splitfun, int threads, bool compute_unique, float splitweight, float distweight, float satweight, 
			float certfactor);
			
%include "librf/random_forest.h"
%extend RandomForest{
    void vs(const float* data, int insts, int atts, signed char* votes, int insts3, int ntrees){
        if (insts != insts3){
            PyErr_Format(PyExc_ValueError, "Given array sizes do not match.");
        }
        else{
            $self->votesset(data, insts, atts, votes);
        }
    }
    
    void nn(const float* points, int m1, int atts, int* neighbors, int m2, int nn1, float* dists, int m3, int nn2, int candidates){
     	if ($self->is_super()){
            PyErr_Format(PyExc_ValueError, "Tried to extract nearest neighbors from a supervised random forest.");
        }
    	else if (nn1 != nn2 || m1 != m2 || m2 != m3){
            PyErr_Format(PyExc_ValueError, "Given array sizes do not match.");
        }
        else if (atts != $self->ndims()){
            PyErr_Format(PyExc_ValueError, "Given points array has wrong number of attributes.");
        }
        else{
        	$self->nearest_multi(points, m1, neighbors, dists, nn1, candidates);
        }
    }
    
    void metric(const float* p1, int m1, int d1, const float* p2, int m2, int d2, float* out, int m3){
     	if ($self->is_super()){
            PyErr_Format(PyExc_ValueError, "Tried to use a supervised forest as a metric.");
        }
    	else if (m1 != m2 || m2 != m3 || d1 != d2){
            PyErr_Format(PyExc_ValueError, "Given array sizes do not match.");
        }
        else if (d1 != $self->ndims()){
            PyErr_Format(PyExc_ValueError, "The given point arrays have the wrong number of attributes.");
        }
        else{
        	$self->metric_multi(p1, p2, out, m1);
        }
    }
};

%extend ClosureUtils{
	static closure* swig_pairwise_closure(const int* constraints, int num_cons, int three){
		return ClosureUtils::pairwise_closure(constraints, num_cons);
	}
}

%pythoncode %{
from numpy import zeros, int32

def constraintsFromClosure(cons):
	"""
	Expands a constraint set by computing its closure and generating the full set of inferable constraints
	
	cons = numconstraints by 3 int32 numpy array containing a must-link or cannot-link point pair on each line
	
	returns: an m by 3 int32 numpy array containing a must-link or cannot-link point pair on each line, where
		m is the number of unique constraints extracted from the computed constraint closure
	"""
	
	if(cons==None or cons.shape[0]==0):
	    return cons
	close = ClosureUtils_swig_pairwise_closure(cons)
	cs = ClosureUtils_constraints_from_closure(close)
	out = zeros((cs.size(), 3), dtype=int32)
	cs.to_array(out)
	del close
	del cs
	return out
%}