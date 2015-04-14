/**
 * @file 
 * @brief A single decision tree
 * The implementation is a port of the Fortran implementation. 
 *
 */

#ifndef _TREE_H_
#define _TREE_H_

#ifdef __GNUC__
/* Test for GCC > 2.95 */
#if __GNUC__ > 2 ||               (__GNUC__ == 2 && (__GNUC_MINOR__ > 95))
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else /* __GNUC__ > 2 ... */
#define likely(x)   (x)
#define unlikely(x) (x)
#endif /* __GNUC__ > 2 ... */
#else /* __GNUC__ */
#define likely(x)   (x)
#define unlikely(x) (x)
#endif /* __GNUC__ */

#include "../general/utility.h"
#include "tree_node.h"
#include <iostream>
#include <vector>
//#include <set>
//#include <map>
#include <math.h>
using namespace std;

typedef enum {
	TRIPLET, PAIR, NONE
} ConstraintType;

//Given an input constraint closure and a set of node IDs comprising a subset of
//the nodes referenced in that closure, computes a new closure describing ONLY the
//constraints between the given nodes
static closure* subclosure(const closure* inclosure,
		const unordered_set<unsigned int>* nodeset);

static const int SINGLE = 0;
static const int RC = 1;

class weight_list;

/**
 * @brief
 * A single decision tree
 *
 *
 *
 * Trees can only be created in two ways:
 *  -# load from a saved model
 *  -# grown from a certain bagging of a dataset (either supervised or un/semi-supervised)
 
 */

void handler(int);

class Tree {
public:
	/// Construct a new tree by loading it from a file
	Tree(istream& in, int K, int splitfun);
	/// Construct a tree by training using labeled data
	Tree(const float* data, int atts, int insts, const int* labels,
			weight_list* weights, int K, int F, unsigned int seed,
			int min_size = 1, float min_gain = 0, int splitfun = SINGLE, int C =
					2, vector<vector<float> >* unique_vals = NULL);
	/// Construct a tree by training using unlabeled data plus triplet constraints
	Tree(const float* data, int atts, int insts, const int* constraints,
			int num_cons, weight_list* weights, int K, int F, unsigned int seed,
			ConstraintType contype, int min_size = 1, float min_gain = 0,
			int splitfun = SINGLE, vector<vector<float> >* unique_vals = NULL,
			topclosure* topclosure = NULL, float splitweight = 1.0f,
			float distweight = 1.0f, float satweight = 1.0f, float certfactor =
					1.0f);
	~Tree(); // clean up
	//get classification result from tree
	int predict(const float* point) const;
	int predict_RC(const float* point) const;
	//get regression result (proportion of positive samples in input's leaf node)
	void regress(const float* point, float* out, float &s) const;
	void regress_RC(const float* point, float* out) const;
	//compute the total gain of each variable in this tree
	void variable_gain(float* vars);

	/// Return the accuracy for the training set
	float training_accuracy() const;
	float oob_accuracy() const;

	//Find nearest neighbors of an input point (unsupervised trees only)
	void nearest(const float* const point, int* const neighbors,
			float* const dists, const int num_neighbors);
	//Use tree as a metric to compute hierarchy distance between two points (single split case)
	float metric_distance(const float* point1, const float* point2);
	//Alternate form, used to compute distance between an input point and an element of the forest
	float metric_distance(const float* point1, const int point2);

	//	void variable_importance(vector<float>* scores, unsigned int* seed) const;

	// do all the work -- separated this from constructor to
	// facilitate threading
	void grow();
	void write(ostream& o) const;
	void read(istream& i);

	int get_num_nodes() {
		return terminal_nodes_;
	}

	int C_; //the number of classes used

private:
	// Node marking
	void add_node(uchar depth);
	void add_node_RC(uchar depth);
	void mark_split(tree_node* n, uint32 split_attr, float split_point,
			float gain);
	void mark_split_RC(tree_node_RC* n);

	void build_tree(); //top-level supervised tree learning function
	void build_tree_semi(); //top-level semi-supervised tree learning function
	void build_node(uint32 node_num); //node-level supervised tree learning function
	void build_node_semi(uint32 node_num); ///node-level semi-supervised tree learning function

	void build_tree_RC(); //top-level supervised random combination tree learning function
	void build_tree_semi_RC(); //top-level semi-supervised random combination tree learning function
	void build_node_RC(uint32 node_num); //node-level supervised random combination tree learning function
	void build_node_semi_RC(uint32 node_num); ///node-level semi-supervised random-combination tree learning function

	void postprocess_semi(); //postprocess built semi-supervised tree to learn constraint-based similarity between leaf nodes

	vector<tree_node*> nodes_;
	vector<tree_node_RC*> nodes_RC_;
	uint32 terminal_nodes_;
	uint32 split_nodes_;
	// A single weight list for all of the instances
	weight_list* weight_list_;
	unsigned int K_;
	unsigned int min_size_;
	float min_gain_;
	uint32 num_instances_;
	uint32 num_instances_sampled_; //the actual number of resampled instanced in the root node
	uint32 num_attributes_;
	uint32 unique_insts_; //only in unsupervised, lists number of distinct instances in this tree
	unsigned int rand_seed_;
	//pointers to data
	const vector<vector<float> >* unique_vals_;
	const float* data_;
	const int* labels_;
	const bool labeled_;
	const int num_cons_;
	//number of different combinations or thresholds to use in each split
	const int F_;
	//defines which function should be used to generate new nodes
	const int splitfun_;
	//(unsupervised only) define the type of constraint used by this tree
	const ConstraintType contype_;

	//weights used for various factors in semi-supervised forest learning
	const float splitweight_;
	const float distweight_;
	const float satweight_;
	const float certfactor_;

	vector<float> unique_scores_; //vector for holding current split scores for each element

	topclosure* topclosure_; //top-level pairwise constraint closure for the forest

	//num_nodes by num_nodes array, holds a float for each pair of leaf nodes indicating constraint-based similarity
	//allocated and built by postprocess_semi() method.
	float* leafsims_;
	//measures certainty for each leaf node pair similarity, otherwise same as leafsims_
	float* leafcerts_;

	static float ln(float num) {
		return log(num);
	}

	//utility functions
	static float lnFunc(float num) {
		if (num < 1e-6) {
			return 0.0f;
		} else {
			return num * log(num);
		}
	}
	static float sublnFunc(float num1, float num2) {
		if (num2 < 1e-6) {
			return num1;
		} else {
			return num1 - num2 * ln(num2);
		}
	}
	static float addlnFunc(float num1, float num2) {
		if (num2 < 1e-6) {
			return num1;
		} else {
			return num1 + num2 * log(num2);
		}
	}
};

//struct sublnFunc: binary_function<float, float, float> {
//	float operator()(const float& num1, const float& num2) const {
//		if (num2 < 1e-6)
//			return num1;
//		else
//			return num1 - num2 * log(num2);
//	}
//}

#endif
