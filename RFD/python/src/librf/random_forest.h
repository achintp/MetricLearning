/** 
 * @file 
 * @brief Randomforest interface
 * This is the interface to manage a random forest
 */
#ifndef _RANDOM_FOREST_H_
#define _RANDOM_FOREST_H_

#include "tree.h"
#include "weights.h"
#include "../general/utility.h"
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>
#include <boost/thread/mutex.hpp>
#include "semaphores/Semaphore.h"

using namespace std;

//simple wrapper for storing indefinite number of constraints
class constraint_set {
public:
	//simple constructor
	constraint_set(vector<int>* Cons) :
			cons(Cons) {
	}
	//returns size of expanded constraint set
	int size() {
		return cons->size() / 3;
	}
	//SWIG-accessible method for returning expanded array of constraints
	void to_array(int* pairs, int numpairs, int three) {
		assert(numpairs == size());
		assert(three == 3);
		for (int i = 0; i < numpairs * 3; i++)
			pairs[i] = (*cons)[i];
		delete cons;
	}
private:
	const vector<int>* cons;
};

class ClosureUtils {
public:
	//Computes constraint closure for a set of must-link and cannot-link constraints
	static topclosure* pairwise_closure(const int* pairs, const int numpairs);

	//Turns a pairwise constraint closure into a set of explicit pairwise constraints
	static constraint_set* constraints_from_closure(const closure* close);
};

class Tree;
/**
 * @brief
 * RandomForest class.  Interface for growing random forests from training
 * data or loading a random forest from disk.
 */
class RandomForest {
public:

	/// Empty constructor
	RandomForest();
	/// Constructor for building from arrays
	RandomForest(const float *data, int atts, int insts, const int* labels,
			int insts2, int num_trees, int K, int F, int min_size, int splitfun,
			int threads, bool compute_unique);
	/// Alternate constructor for training a forest using triplet constraints
	/// rather than element labels
	RandomForest(const float *data, int atts, int insts, const int* constraints,
			int num_cons, int three, int num_trees, int K, int F, int min_size,
			int splitfun, int threads, bool compute_unique, float splitweight,
			float distweight, float satweight, float certfactor);
	~RandomForest();
	//method to train a given tree object - thread-safe
	static void traintree(Tree* reftotrain, int treenum, mom::Semaphore* sem);
	/// Method for getting the vote sets for each instance in a set
	void votesset(const float *data, int atts, int insts, signed char *votes);
	//returns regression values for the given input dataset
	void regrset(const float *data, int atts, int insts, float *regs,
			int insts2, int C);
	//method for efficiently regressing large sets of point made by combining
	//'training' and 'testing' points as in RFD
	void rfdregr(const float* data, int insts, int atts, const float* data2,
			int insts2, int atts2, float* dists, int insts12, int insts22,
			bool position);

	//method for getting the code over a set of points
	void rfdcode(const float* data, int insts, int atts, float* codes,
	int insts1, int trsize, bool position);
	//Compute variable importance (in terms of average entropy gain) for features
	void variable_importance(float* vars, int atts); //NYI for RC trees
	//Find nearest neighbors of an input point (un/semisupervised forests only)
	void nearest(const float* point, int* neighbors, float* dists,
			const int num_neighbors, const int num_candidates) const;
	//Find nearest neighbors of multiple input point (un/semisupervised forests only)
	void nearest_multi(const float* points, const int num_points,
			int* neighbors, float* dists, const int num_neighbors,
			const int num_candidates);
	//Compute distance between two input points (un/semi-supervised forests only)
	float metric_distance(const float* p1, const float* p2) const;
	//Compute distance between an input point an an element of the forest (un/semi-supervised forests only)
	float metric_distance(const float* p1, int p2) const;
	//Compute distance between a number of sets of input points
	void metric_multi(const float* p1, const float* p2, float* out,
			const int m) const;
	//Compute an n by m distance matrix between two sets of n and m input points (un/semisupervised forests only)
	void metric_matrix(const float* p1, const int n, const int d1,
			const float* p2, const int m, const int d2, float* out,
			const int n2, const int m2) const;

	/// Load random forest
	void read(const char filename[]);
	void read(istream& i);
	/// Save random forest
	void write(const char filename[]);
	void write(ostream& o);
	//return number of trees in forest
	int ntrees();
	//return dimensionality of forest
	int ndims() {
		return num_atts_;
	}
	//return number of threads to be used in this forest
	int nthreads() {
		return threads_;
	}
	//return number of classes in this forest
	int C() {
		return C_;
	}
	//set the number of threads to be used in this forest
	void setnthreads(int threads) {
		threads_ = threads;
	}
	//return whether or not this forest was supervised
	bool is_super() {
		return super_;
	}
	//return the number of instances in this un/semisupervised forest (or -1 if forest is supervised)
	int num_inst() {
		if (super_)
			return -1;
		else
			return appear_counts_.size();
	}
private:
	//parallel helper method for regrset
	static void gettreeregs(Tree* tree, const float* data, float* regs,
			int atts, int insts, int ntrees, mom::Semaphore* sem,
			boost::mutex* lock, int splitfun);

	// parallel helper method for votesset
	static void gettreevotes(Tree* tree, const float* data,
			signed char* outvotes, int atts, int insts, int treenum, int ntrees,
			mom::Semaphore* sem, int splitfun);

	//parallel helper method for quickly filling kernel matrix
	static void rfdregrhelper(vector<Tree*>& trees, const float* point,
			const float* data2, int insts, int atts, float* dists,
			mom::Semaphore* sem, int splitfun, bool position);

	//parallel helper to get code words
	static void rfdcodehelper(vector<Tree*>& trees, const float* point, 
	int atts, float* codes, mom::Semaphore* sem);

	//parallel helper method for nearest neighbor retrieval
	static void nearest_multi_worker(const float* points, int* neighbors,
			float* dists, const int num_neighbors, const int num_candidates,
			const vector<int>& workerpts, const RandomForest* forest);

	//parallel helper method for metric distance computation
	static void metric_multi_worker(const float* p1, const float* p2,
			float* out, const int m, const vector<int>& workerpts,
			const RandomForest* forest);

	//parallel helper method for metric distance computation
	static void metric_matrix_worker(const float* p1, const float* p2,
			float* out, const int n, const int m, const vector<int>& workerpts,
			const RandomForest* forest);

	vector<vector<float> >* unique_vals_;
	vector<Tree*> trees_; // component trees in the forest
	int K_; // random vars to try per split
	int F_; //number of combinations/thresholds to try per split
	int min_size_; //minimum node size for the trees
	int threads_; //number of parallel threads to use for training or testing
	int num_atts_; //number of attributes of elements in this forest
	int splitfun_; //the type of splitting function to use for generating tree nodes
	bool super_; //marks whether the forest in question is supervised or unsupervised
	unordered_map<int, float> appear_counts_; //stores the number of trees each instance appears in (unsupervised only)
	//vector<pair<float, int> > var_ranking_; // cached var_ranking
	vector<int> class_weights_;
	//the number of classes in the forest (note, class labels are assumed to range
	//from 0 to C-1
	int C_;
};
#endif
