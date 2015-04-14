/**
 * tree_node.h
 * @file
 * @brief structure for a single node in a decision tree 
 */

#ifndef _TREE_NODE_H_
#define _TREE_NODE_H_
#include "../general/utility.h"
#include <unordered_set>
#include <vector>
#include <fstream>
#include <assert.h>
using namespace std;

typedef enum {
	EMPTY, BUILD_ME, TERMINAL, SPLIT
} NodeStatusType;

struct tree_node {
	/// default constructor initializes to garbage known state and does necessary heap allocation
	tree_node(int C) :
			status(EMPTY), label(99), attr(99), split_point(-999.0), entropy(
					-9999.0), left(0), right(0) {
		instances = new vector<unsigned int>;
		distribution = new float[C];
	}
	~tree_node() {
		delete[] distribution;
	}
	NodeStatusType status;
	uchar label;
	unsigned int attr;
	vector<unsigned int>* instances;
	unordered_set<unsigned int>* set1; //used for storing instance sets (in cases where constant-time instance search is required)
	unordered_set<unsigned int>* set2; //used for storing relevant triplet constraints or unsupervised data
	unsigned int parent;
	unsigned int left;
	unsigned int right;
	float entropy;
	float split_point;
	float* distribution;
	float gain;
	uchar depth;

	void write(ostream& o, int C) const;
	void read(istream& i, int C);
};

class tree_node_RC {
public:
	tree_node_RC() :
			status(EMPTY), label(99), entropy(-9999.0), left(0), right(0) {
	}
	tree_node_RC(unsigned int K, int C) :
			status(EMPTY), label(99), entropy(-9999.0), left(0), right(0) {
		atts = new unsigned int[K];
		weights = new float[K];
		distribution = new float[C];
		instances = new vector<unsigned int>;
	}
	~tree_node_RC() {
		delete[] atts;
		delete[] weights;
		delete[] distribution;
	}

	NodeStatusType status;
	uchar label;
	unsigned int* atts;
	float* weights;
	float bias;
	float* distribution;
	vector<unsigned int>* instances;
	unordered_set<unsigned int>* set1; //used for storing instance sets (in cases where constant-time instance search is required)
	unordered_set<unsigned int>* set2; //used for storing relevant constraints, unsupervised data
	closure* localclosure; //used for storing the transitive closure of a constraint set
	unsigned int parent;
	unsigned int left;
	unsigned int right;
	float entropy;
	uchar depth;

	void write(ostream& o, int K, int C) const;
	void read(istream& i, int K, int C);
};
#endif
