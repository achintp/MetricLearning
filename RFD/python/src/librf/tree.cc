/**
 * @file
 * @brief Tree implementation
 * TODO: fix nodes to be contiguous
 * use ncur idea
 * if current node is split
 * last_node +1 (left)
 * last_node +2 (right)
 * ncur+=2
 */
#include "tree.h"
#include "weights.h"
#include <float.h>
#include <limits.h>
#include <numeric>
#include <set>
#include <utility>
#include <cmath>
 #include <string>
//#include <deque>
//#include <set>
//#include <map>
#include <iostream>
//#include <execinfo.h>
//#include <signal.h>

/*
 * Given an input constraint closure and a set of node IDs comprising a subset of
 * the nodes referenced in that closure, computes a new closure describing ONLY the
 * constraints between the given nodes
 *
 * inclosure = pointer to the larger constraint closure, of which we want to compute a
 * 	subclosure
 * nodeset = the set of node IDs describing which nodes to include in our computed
 * 	subclosure
 *
 * returns: a pointer to a new closure object representing the computed subclosure
 */
closure* subclosure(const closure* inclosure,
		const unordered_set<unsigned int>* nodeset) {
	unordered_set<unsigned int>::const_iterator it;
	unordered_map<int, int>::const_iterator it2;
	vector<pair<int, int> >::const_iterator it3;
	unordered_map<int, unordered_set<int> >::const_iterator it4;
	unordered_set<int>::const_iterator temp;
	closure* outclosure = new closure();

	//cout << "testa1" << endl << flush;

	//first iterate through the node subset and populate the setmembership structure
	//of the new closure
	for (it = nodeset->begin(); it != nodeset->end(); it++) {
		it2 = inclosure->setmembership.find(*it);
		if (it2 != inclosure->setmembership.end())
			outclosure->setmembership[*it] = it2->second;
	}

	//now iterate through the new setmembership to build the pointsets structure
	for (it2 = outclosure->setmembership.begin();
			it2 != outclosure->setmembership.end(); it2++) {
		outclosure->pointsets[it2->second].insert(it2->first);
	}

	//cout << "testa2" << endl << flush;

	//now iterate through the old negative link set, and add all
	//still-relevant links to the new one (where a relevant link is one between
	//two point sets that can both still be found in the NEW pointsets structure)
	unordered_set<int> neglinked;
	for (it3 = inclosure->neglinks.begin(); it3 != inclosure->neglinks.end();
			it3++) {
		if (outclosure->pointsets.count(it3->first) != 0
				&& outclosure->pointsets.count(it3->second) != 0) {
			outclosure->neglinks.push_back(*it3);
			neglinked.insert(it3->first);
			neglinked.insert(it3->second);
		}
	}

	//cout << "testa3" << endl << flush;

	//now iterate through the new pointsets, cleaning up by removing any
	//singleton sets that do not have any negative links
	it4 = outclosure->pointsets.begin();
	while (it4 != outclosure->pointsets.end()) {
		if (it4->second.size() == 1 && neglinked.count(it4->first) == 0) {
			temp = it4->second.begin();
			outclosure->setmembership.erase(*temp);
			outclosure->pointsets.erase(it4++);
		} else
			it4++;
	}

	//cout << "testa4" << endl << flush;

	return outclosure;
}

Tree::Tree(istream& in, int K, int splitfun) :
		// also there is no list of weights
		weight_list_(NULL), splitfun_(splitfun), F_(0), K_(K), C_(0), labeled_(
				true), num_cons_(0), splitweight_(0), distweight_(0), satweight_(
				0), certfactor_(0), contype_(TRIPLET) {
	read(in);
}

/*
 * Constructor using raw arrays
 * data = d by n training data matrix
 * labels = n-length vector of training data labels (0 or 1)
 * weights = produced by sampling with replacement, denotes how many copies of
 * 	each instance in data are present in this training run (minimum 0)
 * K = number of variables to try at each split, or number of variables to use
 * 	in each combination if RC algorithm is used
 * F = number of different random split values to try for each variable at each
 * 	split, or number of different sets of combination weights to try, for RC alg
 * min_size = minimum number of instances a node should contain.  Nodes with this
 * 	many instances or fewer will be marked as leaf nodes.
 * min_gain = minimum entropy gain required to split a node, expressed as a fraction
 * 	of the node's current entropy.  If no split is found that yields sufficient gain,
 * 	the node is marked as a leaf node.
 * seed = random seed to use for this tree
 * splitfun = may be either SINGLE or RC.  If SINGLE, then each node is split on
 * 	a threshold value of a single variable.  If RC, then each node is split on a
 * 	hyperplane through a space composed of K random variables.
 * C = the number of classes in the data (class labels are assumed to range from 0
 * 	to C-1)
 */
Tree::Tree(const float* data, int atts, int insts, const int *labels,
		weight_list* weights, int K, int F, uint32 seed, int min_size,
		float min_gain, int splitfun, int C,
		vector<vector<float> >* unique_vals) :
		data_(data), labels_(labels), labeled_(true), num_cons_(0), weight_list_(
				weights), K_(K), min_size_(min_size), min_gain_(min_gain), num_attributes_(
				atts), num_instances_(insts), F_(F), split_nodes_(0), terminal_nodes_(
				0), rand_seed_(seed), splitfun_(splitfun), C_(C), unique_vals_(
				unique_vals), splitweight_(0), distweight_(0), satweight_(0), certfactor_(
				0), contype_(TRIPLET) {
}

/// Alternate constructor using triplet or pairwise constraints in place of data labels
Tree::Tree(const float* data, int atts, int insts, const int* constraints,
		int num_cons, weight_list* weights, int K, int F, uint32 seed,
		ConstraintType contype, int min_size, float min_gain, int splitfun,
		vector<vector<float> >* unique_vals, topclosure* topclosure,
		float splitweight, float distweight, float satweight, float certfactor) :
		data_(data), labels_(constraints), labeled_(false), num_cons_(num_cons), weight_list_(
				weights), K_(K), min_size_(min_size), min_gain_(min_gain), num_attributes_(
				atts), num_instances_(insts), F_(F), split_nodes_(0), terminal_nodes_(
				0), rand_seed_(seed), splitfun_(splitfun), C_(0), unique_vals_(
				unique_vals), topclosure_(topclosure), splitweight_(
				splitweight), distweight_(distweight), satweight_(satweight), certfactor_(
				certfactor), contype_(contype) {
}

//void handler(int sig) {
//	void *array[10];
//	size_t size;
//
//	// get void*'s for all entries on the stack
//	size = backtrace(array, 10);
//
//	// print out all the frames to stderr
//	fprintf(stderr, "Error: signal %d:\n", sig);
//	backtrace_symbols_fd(array, size, 2);
//	exit(1);
//}

/**
 * Do the work of growing the tree - split here based on whether we are using single-decision
 * or random combination algorithm
 */
void Tree::grow() {
	if (splitfun_ == SINGLE) {
		if (labeled_ == true)
			build_tree();
		else
			build_tree_semi();
	} else {
		if (labeled_ == true)
			build_tree_RC();
		else {
			build_tree_semi_RC();
			if (num_cons_ > 0)
				postprocess_semi();
		}

	}
}

Tree::~Tree() {
	if (weight_list_ != NULL)
		delete weight_list_;
	if (splitfun_ == RC) {
		if (!labeled_) {
			delete leafsims_;
			delete leafcerts_;
			for (vector<tree_node_RC*>::const_iterator it = nodes_RC_.begin();
					it != nodes_RC_.end(); it++) {
				delete (*it)->set1;
				delete *it;
			}
		} else {
			for (vector<tree_node_RC*>::const_iterator it = nodes_RC_.begin();
					it != nodes_RC_.end(); it++)
				delete *it;

		}
	} else {
		if (!labeled_) {
			delete leafsims_;
			delete leafcerts_;
			for (vector<tree_node*>::const_iterator it = nodes_.begin();
					it != nodes_.end(); it++) {
				delete (*it)->set1;
				delete *it;
			}
		} else {
			for (vector<tree_node*>::const_iterator it = nodes_.begin();
					it != nodes_.end(); it++)
				delete *it;

		}
	}
}

/** Save a tree to disk
 * Important things to record:
 *    - Nodes (all active nodes need to be written)
 *    - WeightList ? - this seems irrelevant without the instance set
 *    - Statistics? 
 */
void Tree::write(ostream& o) const {
	if (splitfun_ == SINGLE) {
		o << "Tree: " << nodes_.size() << " " << C_ << endl;
		// Loop through active nodes
		for (int i = 0; i < nodes_.size(); ++i) {
			// Write the node number
			o << i << " ";
			nodes_[i]->write(o, C_);
		}
	} else {
		o << "Tree: " << nodes_RC_.size() << " " << C_ << endl;
		// Loop through active nodes
		for (int i = 0; i < nodes_RC_.size(); ++i) {
			// Write the node number
			o << i << " ";
			nodes_RC_[i]->write(o, K_, C_);
		}
	}
}

/**
 * Read the tree from disk
 */
void Tree::read(istream& in) {
	string spacer;
	int num_nodes;
	in >> spacer >> num_nodes >> C_;
	if (splitfun_ == SINGLE) {
		nodes_.resize(num_nodes);
		for (int i = 0; i < num_nodes; ++i) {
			int cur_node;
			in >> cur_node;
			nodes_[cur_node] = new tree_node(C_);
			nodes_[cur_node]->read(in, C_);
		}
	} else {
		nodes_RC_.resize(num_nodes);
		for (int i = 0; i < num_nodes; ++i) {
			int cur_node;
			in >> cur_node;
			nodes_RC_[cur_node] = new tree_node_RC();
			nodes_RC_[cur_node]->read(in, K_, C_);
		}
	}
}

void Tree::build_tree() {
	int built_nodes = 0;
	// set up ROOT NODE
	add_node(0);
	//add all instances to root
	tree_node* root = nodes_[0];
	for (int i = 0; i < weight_list_->size(); i++) {
		for (int j = 0; j < (*weight_list_)[i]; j++)
			root->instances->push_back(i);
	}
	num_instances_sampled_ = root->instances->size();

	//construct the tree
	do {
		build_node(built_nodes);
		built_nodes++;
	} while (built_nodes < nodes_.size());
}

void Tree::build_tree_semi() {
	int built_nodes = 0;
	// set up ROOT NODE
	add_node(0);
	//add all instances to root
	tree_node* root = nodes_[0];
	root->parent = UINT_MAX;
	root->set2 = new unordered_set<uint32>;
	root->set1 = new unordered_set<uint32>;
	for (int i = 0; i < weight_list_->size(); i++) {
		for (int j = 0; j < (*weight_list_)[i]; j++) {
			root->instances->push_back(i);
			root->set1->insert(i);
		}
	}
	num_instances_sampled_ = root->instances->size();
	unique_insts_ = root->set1->size();
	unique_scores_.resize(num_instances_);

	//now iterate through the constraint list and select those
	//that are entirely contained in the root node's instance set
	if (contype_ == TRIPLET) {
		for (int i = 0; i < num_cons_; i++) {
			if (root->set1->count(uint32(labels_[i * 3])) > 0
					&& root->set1->count(uint32(labels_[i * 3 + 1])) > 0
					&& root->set1->count(uint32(labels_[i * 3 + 2])) > 0)
				root->set2->insert(i);
		}
	} else if (contype_ == PAIR) {
		//NEED TO FIX FOR CLOSURE-BASED METHOD
		for (int i = 0; i < num_cons_; i++) {
			if (root->set1->count(uint32(labels_[i * 3])) > 0
					&& root->set1->count(uint32(labels_[i * 3 + 1])) > 0)
				root->set2->insert(i);
		}
	}

	//construct the tree
	do {
		build_node_semi(built_nodes);
		built_nodes++;
	} while (built_nodes < nodes_.size());
}

void Tree::build_tree_RC() {
	int built_nodes = 0;
	// set up ROOT NODE
	add_node_RC(0);
	//insert all instances to root
	tree_node_RC* root = nodes_RC_[0];
	for (int i = 0; i < weight_list_->size(); i++) {
		for (int j = 0; j < (*weight_list_)[i]; j++) {
			root->instances->push_back(i);
		}
	}

	//construct the tree
	do {
		build_node_RC(built_nodes);
		built_nodes++;
	} while (built_nodes < nodes_RC_.size());
}

void Tree::build_tree_semi_RC() {
	int built_nodes = 0;
	// set up ROOT NODE
	add_node_RC(0);
	//add all instances to root
	tree_node_RC* root = nodes_RC_[0];
	root->parent = UINT_MAX;
	root->set1 = new unordered_set<uint32>;
	//root->set2 = new unordered_set<uint32>;
	//root->set3 = new unordered_set<uint32>;
	for (int i = 0; i < weight_list_->size(); i++) {
		for (int j = 0; j < (*weight_list_)[i]; j++) {
			root->instances->push_back(i);
			root->set1->insert(i);
		}
	}
	unique_insts_ = root->set1->size();
	unique_scores_.resize(num_instances_);

	//now iterate through the constraint list and select those
	//that are entirely contained in the root node's instance set
	if (contype_ == TRIPLET) {
		for (int i = 0; i < num_cons_; i++) {
			if (root->set1->count(uint32(labels_[i * 3])) > 0
					&& root->set1->count(uint32(labels_[i * 3 + 1])) > 0
					&& root->set1->count(uint32(labels_[i * 3 + 2])) > 0)
				root->set2->insert(i);
		}
	} else if (contype_ == PAIR) {
		root->localclosure = topclosure_;
//		bool c1, c2;
//		for (int i = 0; i < num_cons_; i++) {
//			c1 = root->set1->count(uint32(labels_[i * 3])) > 0;
//			c2 = root->set1->count(uint32(labels_[i * 3 + 1])) > 0;
//			if (c1 && c2) {
//				root->set2->insert(i);
//				root->set3->insert(i);
//			} else if (c1 || c2)
//				root->set3->insert(i);
//		}
	}

	//construct the tree
	do {
		build_node_semi_RC(built_nodes);
		built_nodes++;
	} while (built_nodes < nodes_RC_.size());
}

void Tree::mark_split(tree_node* n, uint32 split_attr, float split_point,
		float gain) {
	n->status = SPLIT;
	n->attr = split_attr;
	n->split_point = split_point;
	n->gain = gain;
	n->left = nodes_.size(); // last_node + 1 (due to zero indexing)
	n->right = nodes_.size() + 1; // last_node +2
	split_nodes_++;
	//vars_used_.insert(split_attr);
}

void Tree::mark_split_RC(tree_node_RC* n) {
	n->status = SPLIT;
	n->left = nodes_RC_.size(); // last_node + 1 (due to zero indexing)
	n->right = nodes_RC_.size() + 1; // last_node +2
	split_nodes_++;
	//for (int i = 0; i < K_; i++) {
	//vars_used_.insert(n->atts[i]);
	//}
}

void Tree::add_node(uchar depth) {
	tree_node* n = new tree_node(C_);
	n->status = BUILD_ME;
	n->depth = depth;
	nodes_.push_back(n);
}

void Tree::add_node_RC(uchar depth) {
	tree_node_RC* n = new tree_node_RC(K_, C_);
	n->status = BUILD_ME;
	n->depth = depth;
	nodes_RC_.push_back(n);
}

void Tree::build_node(uint32 node_num) {
	//cout << "building node " << node_num << endl << flush;
	assert(node_num < nodes_.size());
	tree_node* n = nodes_[node_num];
	//count labels of points in this node
	int nump = n->instances->size();
	float* labcounts = new float[C_];
	float* labp;
	for (labp = labcounts; labp < labcounts + C_; labp++)
		*labp = 0;
	vector<uint32>::const_iterator it;
	unordered_set<uint32> unique;
	for (it = n->instances->begin(); it != n->instances->end(); it++) {
		labcounts[labels_[*it]]++;
		unique.insert(*it);
	}
	//assign label to this node
	int argmax = -1;
	int max = -1;
	for (int i = 0; i < C_; i++) {
		if (labcounts[i] > max) {
			max = labcounts[i];
			argmax = i;
		}
	}
	n->label = argmax;

	//assign class distribution and entropy for this node
	for (int i = 0; i < C_; i++) {
		labcounts[i] /= nump;
		(n->distribution)[i] = labcounts[i];
	}
	n->entropy = 0;
	for (labp = labcounts; labp < labcounts + C_; labp++) {
		n->entropy -= lnFunc(*labp);
	}

	//test for complete purity
	bool pure = false;
	for (labp = labcounts; labp != labcounts + C_; labp++) {
		if (*labp > float(nump - 1))
			pure = true;
	}

	//if completely pure or min size, declare terminal
	if (unique.size() <= min_size_ || pure) {
		// cout << "Pure node at depth: " << int(n->depth) << endl;
		// cout << "with node size " << n->instances->size() << endl;
		n->status = TERMINAL;
		delete n->instances;
		terminal_nodes_++;
		return;
	}

	//now find best split
	vector<int> attrs;
	vector<int> split_insts;
	float* split_points = new float[K_];
	float* split_gains = new float[K_];
	float* temp_points = new float[F_];
	float* temp_gains = new float[F_];

	vector < vector<float> > sums; //child by class
	sums.resize(2);
	float sumleft, sumright, best, finsplit, entleft, entright;
	const float* attloc;
	int argbest, i, j, k, finatt;

	//get attribute candidates
	random_sample(num_attributes_, K_, &attrs, &rand_seed_);
	//for each candidate attribute get candidate split points and store the best one
	for (i = 0; i < K_; i++) {
		//pick splits
		//if specified, pick from among only the unique values of this variable,
		//otherwise randomly pick points from this node and use their values
		split_insts.clear();
		if (unique_vals_ != NULL) {
			k = (*unique_vals_)[attrs[i]].size();
			random_sample(k, F_, &split_insts, &rand_seed_);

			for (j = 0; j < split_insts.size(); j++) {
				temp_points[j] = (*unique_vals_)[attrs[i]][split_insts[j]];
			}
		} else {
			random_sample(nump, F_, &split_insts, &rand_seed_);

			for (j = 0; j < split_insts.size(); j++) {
				temp_points[j] = data_[(*(n->instances))[split_insts[j]]
						+ attrs[i] * num_instances_];
			}
		}

		for (j = 0; j < split_insts.size(); j++) {
			//now divide up points according to the chosen attribute and threshold
			sums[0].assign(C_, 0);
			sums[1].assign(C_, 0);
			attloc = data_ + attrs[i] * num_instances_;
			for (it = n->instances->begin(); it != n->instances->end(); it++) {
				sums[byte(attloc[*it] <= temp_points[j])][labels_[*it]]++;
			}
			sumleft = 0;
			sumright = 0;
			for (k = 0; k < C_; k++) {
				sumleft += sums[0][k];
				sumright += sums[1][k];
			}

			//make sure each child node is getting at least one element
			if (sumright >= FLT_EPSILON && sumleft >= FLT_EPSILON) {
				//compute gain for the split
				for (k = 0; k < C_; k++) {
					sums[0][k] /= sumleft;
					sums[1][k] /= sumright;
				}
				entleft = 0;
				entright = 0;
				for (k = 0; k < C_; k++) {
					entleft -= lnFunc(sums[0][k]);
					entright -= lnFunc(sums[1][k]);
				}
				entleft *= sumleft / float(nump);
				entright *= sumright / float(nump);
				temp_gains[j] = n->entropy - entleft - entright;
			} else {
				temp_gains[j] = 0;
			}
		}
		//now find the best of the tested splits
		best = temp_gains[0];
		argbest = 0;
		for (j = 1; j < split_insts.size(); j++) {
			if (temp_gains[j] > best) {
				best = temp_gains[j];
				argbest = j;
			}
		}
		split_points[i] = temp_points[argbest];
		split_gains[i] = best;
	}

	//now go through the best splits found for each attribute, and pick the best
	//attribute to split on
	best = split_gains[0];
	argbest = 0;
	for (i = 1; i < K_; i++) {
		if (split_gains[i] > best) {
			best = split_gains[i];
			argbest = i;
		}
	}
	finsplit = split_points[argbest];
	finatt = attrs[argbest];
	delete[] split_points, split_gains, temp_points, temp_gains;



	//check to see if we have met minimum gain requirement
	if (best / n->entropy < min_gain_ || best < FLT_EPSILON) {
		cout << "Best " << best << endl;
		n->status = TERMINAL;
		delete n->instances;
		terminal_nodes_++;
		return;
	}

	//mark as split and create child nodes
	mark_split(n, finatt, finsplit, best);
	add_node(n->depth + 1);
	add_node(n->depth + 1);
	tree_node* left = nodes_[n->left];
	tree_node* right = nodes_[n->right];

	//split this node's instances between child nodes
	attloc = data_ + finatt * num_instances_;
	for (it = n->instances->begin(); it != n->instances->end(); it++) {
		if (attloc[*it] <= finsplit) {
			left->instances->push_back(*it);
		} else {
			right->instances->push_back(*it);
		}
	}
	//clean up no-longer-needed variables
	delete n->instances;
}

void Tree::build_node_semi(uint32 node_num) {
	float sumleft, sumright, best, finsplit, temp, temp2, meandist, spliteven,
			constraintsat, possat, negsat, postot, negtot;
	const float* attloc;
	int argbest, i, j, k, finatt;
	//cout << "building node " << node_num << endl << flush;
	assert(node_num < nodes_.size());
	tree_node* n = nodes_[node_num];
	int nump = n->instances->size();
	vector<uint32>::const_iterator it;
	unordered_multimap<uint32, float>::const_iterator it2;
	unordered_set<uint32>::const_iterator it3;

	//if number of unique items is at or below min size, declare terminal
	if (n->set1->size() <= min_size_) {
		//cout << "Pure node at depth: " << int(n->depth) << endl;
		//cout << "with node size " << n->instances->size() << endl;
		n->status = TERMINAL;
		delete n->instances;
		delete n->set2;
		terminal_nodes_++;
		return;
	}

	//now find best split
	vector<int> attrs;
	vector<int> split_insts;
	float* split_points = new float[K_];
	float* split_gains = new float[K_];
	float* temp_points = new float[F_];
	float* temp_scores = new float[F_];

	vector < unordered_multimap<uint32, float> > splitres; //lists elements and distances from split for each child
	splitres.resize(2);

	//get attribute candidates
	random_sample(num_attributes_, K_, &attrs, &rand_seed_);

	//compute local variance in each split candidate dimension
	float* dim_vars = new float[K_];
	for (int i = 0; i < K_; i++) {
		attloc = data_ + attrs[i] * num_instances_;
		temp = 0;
		for (it = n->instances->begin(); it != n->instances->end(); it++) {
			temp += attloc[*it];
		}
		temp = temp / nump;
		dim_vars[i] = 0;
		for (it = n->instances->begin(); it != n->instances->end(); it++) {
			temp2 = attloc[*it] - temp;
			dim_vars[i] += temp2 * temp2;
		}
		dim_vars[i] = sqrt(dim_vars[i] / nump);
	}

	//in case of pairwise constraints, compute total number of positive and negative constraints
	if (contype_ == PAIR) {
		postot = 0;
		negtot = 0;
		for (it3 = n->set2->begin(); it3 != n->set2->end(); it3++) {
			if (labels_[*it3 * 3 + 2] < 0)
				negtot++;
			else
				postot++;
		}
	}

	//for each candidate attribute get candidate split points and store the best one
	for (i = 0; i < K_; i++) {
		//pick splits
		//if specified, pick from among only the unique values of this variable,
		//otherwise randomly pick points from this node and use their values
		split_insts.clear();
		if (unique_vals_ != NULL) {
			k = (*unique_vals_)[attrs[i]].size();
			random_sample(k, F_, &split_insts, &rand_seed_);

			for (j = 0; j < split_insts.size(); j++) {
				temp_points[j] = (*unique_vals_)[attrs[i]][split_insts[j]];
			}
		} else {
			random_sample(nump, F_, &split_insts, &rand_seed_);

			for (j = 0; j < split_insts.size(); j++) {
				temp_points[j] = data_[(*(n->instances))[split_insts[j]]
						+ attrs[i] * num_instances_];
			}
		}

		for (j = 0; j < split_insts.size(); j++) {
			//now divide up points according to the chosen attribute and threshold
			splitres[0].clear();
			splitres[1].clear();
			attloc = data_ + attrs[i] * num_instances_;
			for (it = n->instances->begin(); it != n->instances->end(); it++) {
				temp = attloc[*it] - temp_points[j];
				splitres[byte(temp <= 0)].insert(make_pair(*it, abs(temp)));
			}

			//make sure each child node is getting at least one element
			if (splitres[0].size() > 0 && splitres[1].size() > 0) {
				//compute score for the split
				//first compute distance score via a gaussian on distance with sigma determined by variance in this dimension
				temp = dim_vars[i] / 4;
				for (it2 = splitres[0].begin(); it2 != splitres[0].end(); it2++)
					meandist += 1 - exp(-pow(it2->second, 2) / temp);
				for (it2 = splitres[1].begin(); it2 != splitres[1].end(); it2++)
					meandist += 1 - exp(-pow(it2->second, 2) / temp);
				meandist = meandist / float(nump);
				//now determine evenness of split (i.e. select against unbalanced splits)
				spliteven = float(min(splitres[0].size(), splitres[1].size()))
						/ float(nump);
				//finally determine proportion of satisfied constraints
				//(interpreting relevant dual triplet constraints as must-link constraints)
				constraintsat = 0;

				if (n->set2->size() > 0) {
					//triplet constraints
					if (contype_ == TRIPLET) {
						for (it3 = n->set2->begin(); it3 != n->set2->end();
								it3++) {
							constraintsat +=
									byte(
											(splitres[0].count(
													uint32(labels_[*it3 * 3]))
													> 0)
													== (splitres[0].count(
															uint32(
																	labels_[*it3
																			* 3
																			+ 1]))
															> 0));
						}
						constraintsat /= float(n->set2->size());
					}
					//pairwise constraints
					else {
						for (it3 = n->set2->begin(); it3 != n->set2->end();
								it3++) {
							if (splitres[0].count(uint32(labels_[*it3 * 3]))
									== splitres[0].count(
											uint32(labels_[*it3 * 3 + 1]))) {
								if (labels_[*it3 * 3 + 2] >= 0)
									possat++;
							} else if (labels_[*it3 * 3 + 2] < 0)
								negsat++;
						}
						possat /= postot;
						negsat /= negtot;
						constraintsat = possat * negsat;
					}

					//combine these three measures using a weighted product to compute score
					temp_scores[j] = pow(spliteven, splitweight_)
							/ pow(meandist, distweight_)
							* pow(constraintsat, satweight_);
				} else {
					//if no constraints, use only distance and evenness to compute score
					temp_scores[j] = pow(spliteven, splitweight_)
							/ pow(meandist, distweight_);
				}
			} else {
				temp_scores[j] = 0;
			}
		}
		//now find the best of the tested splits
		best = temp_scores[0];
		argbest = 0;
		for (j = 1; j < split_insts.size(); j++) {
			if (temp_scores[j] > best) {
				best = temp_scores[j];
				argbest = j;
			}
		}
		split_points[i] = temp_points[argbest];
		split_gains[i] = best;
	}

	//now go through the best splits found for each attribute, and pick the best
	//attribute to split on
	best = split_gains[0];
	argbest = 0;
	for (i = 1; i < K_; i++) {
		if (split_gains[i] > best) {
			best = split_gains[i];
			argbest = i;
		}
	}
	finsplit = split_points[argbest];
	finatt = attrs[argbest];
	delete[] split_points, split_gains, temp_points, temp_scores, dim_vars;

	//mark as split and create child nodes
	mark_split(n, finatt, finsplit, best);
	add_node(n->depth + 1);
	add_node(n->depth + 1);
	tree_node* left = nodes_[n->left];
	tree_node* right = nodes_[n->right];
	left->parent = node_num;
	left->set1 = new unordered_set<unsigned int>;
	left->set2 = new unordered_set<unsigned int>;
	right->parent = node_num;
	right->set1 = new unordered_set<unsigned int>;
	right->set2 = new unordered_set<unsigned int>;

	//split this node's instances between child nodes
	attloc = data_ + finatt * num_instances_;
	for (it = n->instances->begin(); it != n->instances->end(); it++) {
		if (attloc[*it] <= finsplit) {
			left->instances->push_back(*it);
			left->set1->insert(*it);
		} else {
			right->instances->push_back(*it);
			right->set1->insert(*it);
		}
	}

	//determine relevant triplet constraints in child nodes
	//iterate through parent node's relevant constraints and select those
	//that are entirely contained in each child's instance set
	for (it3 = n->set2->begin(); it3 != n->set2->end(); it3++) {
		if (left->set1->count(uint32(labels_[*it3 * 3])) > 0
				&& left->set1->count(uint32(labels_[*it3 * 3 + 1])) > 0
				&& left->set1->count(uint32(labels_[*it3 * 3 + 2])) > 0)
			left->set2->insert(*it3);
		else if (right->set1->count(uint32(labels_[*it3 * 3])) > 0
				&& right->set1->count(uint32(labels_[*it3 * 3 + 1])) > 0
				&& right->set1->count(uint32(labels_[*it3 * 3 + 2])) > 0)
			right->set2->insert(*it3);
	}

	delete n->instances;
	delete n->set2;
}

void Tree::build_node_RC(uint32 node_num) {
	assert(node_num < nodes_RC_.size());
	tree_node_RC* n = nodes_RC_[node_num];
	//count labels of points in this node
	int nump = n->instances->size();
	float* labcounts = new float[C_];
	float* labp;
	for (labp = labcounts; labp < labcounts + C_; labp++)
		*labp = 0;
	vector<uint32>::const_iterator it;
	unordered_set<uint32> unique;
	for (it = n->instances->begin(); it != n->instances->end(); it++) {
		labcounts[labels_[*it]]++;
		unique.insert(*it);
	}
	//assign label to this node
	int argmax = -1;
	int max = -1;
	for (int i = 0; i < C_; i++) {
		if (labcounts[i] > max) {
			max = labcounts[i];
			argmax = i;
		}
	}
	n->label = argmax;

	//test for complete purity
	bool pure = false;
	for (labp = labcounts; labp != labcounts + C_; labp++) {
		if (*labp > float(nump - 1))
			pure = true;
	}

	//assign class distribution and entropy for this node
	for (int i = 0; i < C_; i++) {
		labcounts[i] /= nump;
		(n->distribution)[i] = labcounts[i];
	}
	n->entropy = 0;
	for (labp = labcounts; labp < labcounts + C_; labp++) {
		n->entropy -= lnFunc(*labp);
	}

	//if completely pure or min size, declare terminal
	if (unique.size() <= min_size_ || pure) {
		//cout << "Pure node at depth: " << int(n->depth) << endl;
		//cout << "with node size " << n->instances->size() << endl;
		n->status = TERMINAL;
		delete n->instances;
		terminal_nodes_++;
		return;
	}

	//now begin process of finding best split
	vector<int> attrs;
	vector < vector<float> > combs;
	combs.resize(F_);
	vector<float> biases;
	biases.resize(F_);
	float score, entleft, entright;
	int i, j, k, inst, split;
	vector<float> gains;
	vector<float> scores;
	scores.resize(nump);
	gains.resize(F_);
	//loop until we find a split that actually subdivides the data
	bool good = false;
	while (!good) {
		//randomly select K unique attributes to split on
		attrs.clear();
		random_sample(num_attributes_, K_, &attrs, &rand_seed_);
		for (i = 0; i < K_; i++) {
			n->atts[i] = uint32(attrs[i]);
		}
		//generate F random sets of weights for the selected attributes
		for (i = 0; i < F_; i++) {
			combs[i].resize(K_);
			for (int j = 0; j < K_; j++) {
				combs[i][j] = rand_float(-1, 1, &rand_seed_);
			}
		}
		//find the distribution of the data for each of the generated weight sets
		//(i.e. number of instances of each class in each child node for each split)
		vector < vector<vector<float> > > sums; //split by child by class
		sums.resize(F_);
		for (i = 0; i < F_; i++) {
			//2 = number of child nodes
			sums[i].resize(2);
			sums[i][0].assign(C_, 0);
			sums[i][1].assign(C_, 0);
		}
		for (j = 0; j < F_; j++) {
			for (i = 0; i < nump; i++) {
				inst = (*(n->instances))[i];
				score = 0;
				for (k = 0; k < K_; k++) {
					score += combs[j][k]
							* data_[inst + (attrs[k]) * num_instances_];
				}
				scores[i] = score;
			}
			biases[j] = scores[rand_r(&rand_seed_) % nump];

			for (i = 0; i < nump; i++) {
				split = int((scores[i] > biases[j]));
				sums[j][split][labels_[(*(n->instances))[i]]]++;
			}
		}

		//compute gains
		float sumright, sumleft;
		for (j = 0; j < F_; j++) {
			//find child node sizes
			sumleft = 0;
			sumright = 0;
			for (k = 0; k < C_; k++) {
				sumleft += sums[j][0][k];
				sumright += sums[j][1][k];
			}

			//make sure each child node is getting at least one element
			if (sumright >= FLT_EPSILON && sumleft >= FLT_EPSILON) {
				//compute gain for the split
				for (i = 0; i < C_; i++) {
					sums[j][0][i] /= sumleft;
					sums[j][1][i] /= sumright;
				}

				entleft = 0;
				entright = 0;
				for (k = 0; k < C_; k++) {
					entleft -= lnFunc(sums[j][0][k]);
					entright -= lnFunc(sums[j][1][k]);
				}
				entleft *= sumleft / float(nump);
				entright *= sumright / float(nump);
				gains[j] = n->entropy - entleft - entright;

				if (gains[j] > FLT_EPSILON)
					good = true;
			} else {
				gains[j] = 0;
			}
		}
	}
	//find best split
	float best = 0;
	int argbest = -1;
	for (i = 0; i < F_; i++) {
		if (gains[i] > best) {
			best = gains[i];
			argbest = i;
		}
	}

	//check to see if we have met minimum gain requirement
	if (best / n->entropy < min_gain_) {
		n->status = TERMINAL;
		delete n->instances;
		terminal_nodes_++;
		return;
	}

	//use the minimum entropy split to set coefficients for this node
	for (i = 0; i < K_; i++) {
		n->weights[i] = combs[argbest][i];
	}
	n->bias = biases[argbest];

	//mark as split and create child nodes
	mark_split_RC(n);
	add_node_RC(n->depth + 1);
	add_node_RC(n->depth + 1);
	tree_node_RC* left = nodes_RC_[n->left];
	tree_node_RC* right = nodes_RC_[n->right];
	//split this node's instances between child nodes
	float* nweights = n->weights;
	for (i = 0; i < nump; i++) {
		inst = (*(n->instances))[i];
		score = 0;
		for (j = 0; j < K_; j++) {
			score += nweights[j] * data_[inst + (attrs[j]) * num_instances_];
		}
		if (score <= n->bias) {
			left->instances->push_back(inst);
		} else {
			right->instances->push_back(inst);
		}
	}
	//clean up no-longer-needed instance list for this node
	delete n->instances;
}

void Tree::build_node_semi_RC(uint32 node_num) {
	assert(node_num < nodes_RC_.size());
	//declare vars
	float sumleft, sumright, best, finsplit, temp, temp2, meandist, meanval,
			spliteven, csetrat, posentropy, negsat, constraintgain, nsize,
			sizel, sizer, tempnorm;
	const float* attloc;
	bool c1, c2;
	int argbest, i, j, k, finatt, split;
	tree_node_RC* n = nodes_RC_[node_num];
	int nump = n->instances->size();
	nsize = float(nump) / num_instances_sampled_;
	vector<uint32>::const_iterator it;
	unordered_multimap<uint32, float>::const_iterator it2;
	unordered_set<uint32>::const_iterator it3;
	vector<int> attrs;
	vector < vector<float> > combs;
	vector<float> biases, scores;
	float score, entleft, entright;
	vector < unordered_multimap<uint32, float> > splitres; //lists elements and distances from split for each child
	unordered_map<int, pair<float, float> > setrats; //<setid, <setsize, splitrat> >
	pair<float, float> tempset1, tempset2;
	unordered_map<int, pair<float, float> >::const_iterator setratit;
	unordered_map<int, unordered_set<int> >::const_iterator csetit;
	unordered_set<int>::const_iterator csetit2;
	vector<pair<int, int> >::const_iterator neglinkit;

	//cout << "building node " << node_num << endl << flush;

	//if number of unique items is at or below min size, declare terminal
	if (n->set1->size() <= min_size_) {
		//cout << "Pure node at depth: " << int(n->depth) << endl;
		//cout << "with node size " << n->set1->size() << endl;
		n->status = TERMINAL;
		delete n->instances;
		if (node_num != 0 && contype_ == PAIR)
			delete n->localclosure;
		//delete n->set2;
		terminal_nodes_++;
		return;
	}

	//now begin process of finding best split
	float* temp_scores = new float[F_];
	splitres.resize(2);

	combs.resize(F_);
	biases.resize(F_);
	scores.resize(nump);

	//loop until we find a split that actually subdivides the data
	bool good = false;
	int trycount = 0;
	while (!good) {
		//randomly select K unique attributes to split on
		attrs.clear();
		random_sample(num_attributes_, K_, &attrs, &rand_seed_);
		for (i = 0; i < K_; i++)
			n->atts[i] = uint32(attrs[i]);

		//compute harmonic mean local value of these attributes
		/*float* meanvals = new float[K_];
		 for (int i = 0; i < K_; i++) {
		 attloc = data_ + attrs[i] * num_instances_;
		 temp = 0;
		 for (it = n->instances->begin(); it != n->instances->end(); it++) {
		 temp += attloc[*it];
		 }
		 meanvals[i] = temp / nump;
		 }
		 meanval = 0;
		 for (int i = 0; i < K_; i++)
		 meanval += 1 / meanvals[i];
		 meanval = float(K_) / meanval;
		 temp = abs(meanval / 10);

		 delete[] meanvals;*/
		temp = 0.1;

		//generate F random sets of weights for the selected attributes
		for (i = 0; i < F_; i++) {
			combs[i].resize(K_);
			tempnorm = 0.0f;
			for (int j = 0; j < K_; j++) {
				combs[i][j] = rand_float(-1, 1, &rand_seed_);
				tempnorm += combs[i][j] * combs[i][j];
			}
			tempnorm = sqrt(tempnorm);
			for (int j = 0; j < K_; j++)
				combs[i][j] = combs[i][j] / tempnorm;
		}

		//outer loop to compute gains for each set of weights
		for (j = 0; j < F_; j++) {
			//find split result
			splitres[0].clear();
			splitres[1].clear();
			//compute split score position/score for each unique element in the node
			for (it3 = n->set1->begin(); it3 != n->set1->end(); it3++) {
				score = 0;
				attloc = data_ + *it3;
				for (k = 0; k < K_; k++)
					score += combs[j][k] * attloc[(attrs[k]) * num_instances_];
				unique_scores_[*it3] = score;
			}
			//use the computed scores from each unique element to fill the scores for each
			//actual (potentially non-unique) instance in the node
			for (i = 0; i < nump; i++)
				scores[i] = unique_scores_[(*(n->instances))[i]];

			//randomly pick a few values and average them to get our bias
			for (i = 0; i < K_; i++)
				biases[j] += scores[rand_r(&rand_seed_) % nump];
			biases[j] /= K_;

			for (i = 0; i < nump; i++) {
				split = int((scores[i] > biases[j]));
				splitres[split].insert(
						make_pair((*(n->instances))[i],
								abs(score - biases[j])));
			}

			//make sure each child node is getting at least one element
			if (splitres[0].size() > 0 && splitres[1].size() > 0) {
				sizel = splitres[0].size() / float(nump);
				sizer = splitres[1].size() / float(nump);

				//compute gain for the split
				//first use the previously computed harmonic mean of local feature values to set the sigma for a gaussian
				//kernel on the distance from the split point along the projection axis defined by this discriminant
				meandist = 0;
				for (it2 = splitres[0].begin(); it2 != splitres[0].end(); it2++)
					meandist += 1 - exp(-pow(it2->second, 2) / temp);
				for (it2 = splitres[1].begin(); it2 != splitres[1].end(); it2++)
					meandist += 1 - exp(-pow(it2->second, 2) / temp);
				meandist = meandist / float(nump);

				//now determine evenness of split (i.e. select against unbalanced splits)
				spliteven = min(sizel, sizer);

				//finally determine constraint satisfaction in each child node, and compute increase/decrease
				//in constraint satisfaction relative to parent
				//constraintsatl = 0;
				//constraintsatr = 0;
				//triplet constraints
				//if (contype_ == TRIPLET) {
//						for (it3 = n->set2->begin(); it3 != n->set2->end();
//								it3++) {
//							constraintsat +=
//									byte(
//											(splitres[0].count(
//													uint32(labels_[*it3 * 3]))
//													> 0)
//													== (splitres[0].count(
//															uint32(
//																	labels_[*it3
//																			* 3
//																			+ 1]))
//															> 0));
//						}
//						constraintsat /= float(n->set2->size());
				//}
				//pairwise constraints
				if (contype_ == PAIR) {
					if (n->localclosure->pointsets.size() > 0) {
						//first determine split ratio (i.e. proportion of member nodes going to the right)
						//for each constrained point set in the closure
						for (csetit = n->localclosure->pointsets.begin();
								csetit != n->localclosure->pointsets.end();
								csetit++) {
							csetrat = 0.0f;
							for (csetit2 = csetit->second.begin();
									csetit2 != csetit->second.end();
									csetit2++) {
								if (unique_scores_[*csetit2] > biases[j])
									csetrat += 1.0f;
							}
							setrats[csetit->first] = make_pair(
									csetit->second.size(),
									csetrat / csetit->second.size());
						}
						//now, for must-link constraints, compute an entropy value over all the constraint ratios:
						// entropy = -sum_i(setsize_i*(ratio_i*log(ratio_i) + (1-ratio_i)*log(1-ratio_i))
						posentropy = 0.0f;
						for (setratit = setrats.begin();
								setratit != setrats.end(); setratit++) {
							csetrat = setratit->second.second;
							posentropy -= (setratit->second.first)
									* (csetrat * lnFunc(csetrat)
											+ (1 - csetrat)
													* lnFunc(1 - csetrat));
						}
						posentropy /= n->set1->size();
						//Now compute cannot-link constraint satisfaction.  In this case, we want to ensure that
						//negatively-linked csets are being split largely in different directions, so we compute
						//negative satisfaction as sum_k(min(size_k1,size_k2)*|ratio_k1 - ratio_k2|)
						if (n->localclosure->neglinks.size() > 0) {
							negsat = 0.0f;
							for (neglinkit = n->localclosure->neglinks.begin();
									neglinkit != n->localclosure->neglinks.end();
									neglinkit++) {
								tempset1 = setrats[neglinkit->first];
								tempset2 = setrats[neglinkit->second];
								negsat += min(tempset1.first, tempset2.first)
										* abs(
												tempset1.second
														- tempset2.second);
							}
							negsat /= n->set1->size();
						} else
							negsat = 1.0f;

						//cout << "ent/sat " << posentropy << " " << negsat
						//<< endl << flush;

						//now compute overall constraint satisfaction.  We want to enforce low positive entropy
						//and high negative satisfaction, so we compute overall satisfaction as negsat/posentropy.
						constraintgain = negsat / posentropy;
					} else
						constraintgain = -1.0f;
				} else if (contype_ == NONE)
					constraintgain = -1.0f;

				//combine these three measures using a weighted product to compute score
				if (constraintgain > 0)
					temp_scores[j] = pow(spliteven, splitweight_)
							/ pow(meandist, distweight_)
							* pow(constraintgain, satweight_);
				//if no constraints, use only distance and evenness to compute score
				else
					temp_scores[j] = pow(spliteven, splitweight_)
							/ pow(meandist, distweight_);
				//cout << "split/dist/constraint/totscore " << spliteven << " "
				//		<< meandist << " " << constraintgain << " "
				//		<< temp_scores[j] << endl << flush;
				if (temp_scores[j] > 0)
					good = true;
			} else {
				temp_scores[j] = 0;
			}
		}
		trycount++;
		//if there appears to be no possible good split for this node, declare it terminal
		if (trycount > 5 && !good) {
			cout << "nogood " << n->set1->size() << " " << spliteven << " "
					<< meandist << " " << constraintgain << " "
					<< temp_scores[j] << endl << flush;
			n->status = TERMINAL;
			delete n->instances;
			delete n->localclosure;
			terminal_nodes_++;
			return;
		}
	}

	//cout << "test1" << endl << flush;

	//find best split
	best = 0;
	argbest = -1;
	for (i = 0; i < F_; i++) {
		if (temp_scores[i] > best) {
			best = temp_scores[i];
			argbest = i;
		}
	}

	delete[] temp_scores;

	//mark as split and create child nodes
	mark_split_RC(n);
	for (int i = 0; i < K_; i++)
		n->weights[i] = combs[argbest][i];
	n->bias = biases[argbest];

	//cout << "test2" << endl << flush;

	add_node_RC(n->depth + 1);
	add_node_RC(n->depth + 1);
	tree_node_RC* left = nodes_RC_[n->left];
	tree_node_RC* right = nodes_RC_[n->right];
	left->parent = node_num;
	left->set1 = new unordered_set<unsigned int>;
	//left->set2 = new unordered_set<unsigned int>;
	//left->set3 = new unordered_set<unsigned int>;
	right->parent = node_num;
	right->set1 = new unordered_set<unsigned int>;
	//right->set2 = new unordered_set<unsigned int>;
	//right->set3 = new unordered_set<unsigned int>;

	//split this node's instances between child nodes
	float* nweights = n->weights;
	for (it = n->instances->begin(); it != n->instances->end(); it++) {
		score = 0;
		for (j = 0; j < K_; j++)
			score += nweights[j] * data_[*it + (attrs[j]) * num_instances_];
		if (score <= n->bias) {
			left->instances->push_back(*it);
			left->set1->insert(*it);
		} else {
			right->instances->push_back(*it);
			right->set1->insert(*it);
		}
	}

	//cout << "test3" << endl << flush;

	//determine relevant constraints in child nodes
	//iterate through parent node's relevant constraints and select those
	//that are entirely contained in each child's instance set
	if (contype_ == TRIPLET) {
		for (it3 = n->set2->begin(); it3 != n->set2->end(); it3++) {
			if (left->set1->count(uint32(labels_[*it3 * 3])) > 0
					&& left->set1->count(uint32(labels_[*it3 * 3 + 1])) > 0
					&& left->set1->count(uint32(labels_[*it3 * 3 + 2])) > 0)
				left->set2->insert(*it3);
			else if (right->set1->count(uint32(labels_[*it3 * 3])) > 0
					&& right->set1->count(uint32(labels_[*it3 * 3 + 1])) > 0
					&& right->set1->count(uint32(labels_[*it3 * 3 + 2])) > 0)
				right->set2->insert(*it3);
		}
	} else if (contype_ == PAIR) {
		left->localclosure = subclosure(n->localclosure, left->set1);
		right->localclosure = subclosure(n->localclosure, right->set1);
	}

	delete n->instances;
	//delete the closure info for this node, unless it is root (since root holds reusable master closure)
	if (node_num != 0 && contype_ == PAIR)
		delete n->localclosure;
	//delete n->set2;
	//delete n->set3;
	//cout << "test4" << endl << flush;
}

/*
 * Postprocesses a learned semi-supervised tree to learn a constraint-based similarity value
 * 	(plus associated certainty factor) between each pair of leaf nodes.
 * 	note: similarity ranges from 0 to 1, and is actually a disimilarity measure (to be consistent
 * 		with the tree-based distance measure), so a 0 is maximally similar
 */
void Tree::postprocess_semi() {
	int numnodes, i, j, reltot, knowntot, set1val, set2val;
	float denom;
	unordered_map<int, int>::iterator set1, set2;
	vector<uint32>::const_iterator it1, it2;
	unordered_set<uint32>::const_iterator it3, it4;
	vector<uint32> leaflist;
	if (splitfun_ == RC) {
		//initialize
		numnodes = nodes_RC_.size();
		tree_node_RC* n1;
		tree_node_RC* n2;
		leafsims_ = new float[numnodes * numnodes];
		leafcerts_ = new float[numnodes * numnodes];
		//build list of leaf nodes
		for (i = 0; i != nodes_RC_.size(); i++) {
			if (nodes_RC_[i]->status == TERMINAL)
				leaflist.push_back(i);
		}
		//now iterate through each leaf node pair and compute similarity and certainty
		for (it1 = leaflist.begin(); it1 != leaflist.end(); it1++) {
			n1 = nodes_RC_[*it1];
			for (it2 = it1; it2 != leaflist.end(); it2++) {
				n2 = nodes_RC_[*it2];
				reltot = 0;
				knowntot = 0;
				for (it3 = n1->set1->begin(); it3 != n1->set1->end(); it3++) {
					set1 = topclosure_->setmembership.find(*it3);
					for (it4 = n2->set1->begin(); it4 != n2->set1->end();
							it4++) {
						set2 = topclosure_->setmembership.find(*it4);
						//now in innermost loop, check constraint status for each element pair
						//and use to increment/decrement reltot and knowntot
						if (set1 != topclosure_->setmembership.end()
								&& set2 != topclosure_->setmembership.end()) {
							set1val = set1->second;
							set2val = set2->second;
							//check for must-link
							if (set1val == set2val) {
								reltot--;
								knowntot++;
							} else {
								//ensure pair follows (lesser, greater) convention
								if (set2val < set1val)
									swap(set1val, set2val);
								//check for cannot-link
								if (topclosure_->neglinkset.find(
										make_pair(set1val, set2val))
										!= topclosure_->neglinkset.end()) {
									reltot++;
									knowntot++;
								}
							}
						}
					}
				}
				//now that we have reltot and knowntot for this leaf-node pair, compute similarity and certainty
				denom = (n1->set1->size() * n2->set1->size());
				//similarity = 1 + (total known relationship score)/(2 * total number of element pairs)
				leafsims_[*it1 + *it2 * numnodes] = 0.5f
						+ (float(reltot) / (2.0f * denom));
				leafsims_[*it2 + *it1 * numnodes] = leafsims_[*it1
						+ *it2 * numnodes];
				//certainty = certfactor * (total number of known relationships)/(total number of element pairs)    (max 1)
				leafcerts_[*it1 + *it2 * numnodes] = min(
						certfactor_ * float(knowntot) / denom, 1.0f);
				leafcerts_[*it2 + *it1 * numnodes] = leafcerts_[*it1
						+ *it2 * numnodes];
			}
		}
	} else {
		numnodes = nodes_.size();
		//NOT YET IMPLEMENTED
	}

}

/*
 * @param point d-length float array containing the instance whose class we want to predict
 */
int Tree::predict(const float* point) const {
	//base case
	bool result = false;
	int cur_node = 0;
	unsigned char label = 0;
	while (!result) {
		const tree_node* n = nodes_[cur_node];
		if (n->status == TERMINAL) {
			result = true;
			label = n->label;
		} else {
			if (point[n->attr] <= n->split_point) {
				cout << "0";
				cur_node = n->left;
			} else {
				cout << "1";
				cur_node = n->right;
			}
		}
	}
	return int(label);
}

int Tree::predict_RC(const float* point) const {
	//base case
	bool result = false;
	uint32* atts;
	float* weights;
	float score;
	int i;
	int cur_node = 0;
	unsigned char label = 0;
	while (!result) {
		const tree_node_RC* n = nodes_RC_[cur_node];
		if (n->status == TERMINAL) {
			result = true;
			label = n->label;
		} else {
			atts = n->atts;
			weights = n->weights;
			score = 0;
			for (i = 0; i < K_; i++) {
				score += point[atts[i]] * weights[i];
			}
			if (score <= n->bias) {
				cur_node = n->left;
			} else {
				cur_node = n->right;
			}
		}
	}
	return int(label);
}

/*
 * @param point d-length float array containing the instance to regress
 */
void Tree::regress(const float* point, double &code) const {
	//base case
	const tree_node* n;
	int cur_node = 0;
	code = 1;
	// cout << "In regress" << endl;
	while (true) {
		n = nodes_[cur_node];
		if (n->status == TERMINAL) {
			for (int i = 0; i < C_; i++){
				// out[i] = (n->distribution)[i];
			}
			// cout << C_ << endl;
			// cout << s << endl;
			break;
		} else {
			if (point[n->attr] <= n->split_point) {
				code *= 10;
				cur_node = n->left;
			} else {
				code *= 10;
				code += 1;
				cur_node = n->right;
			}
		}
	}
}

/*
 * @param point d-length float array containing the instance to regress
 */
void Tree::regress_RC(const float* point, float* out) const {
	//base case
	uint32* atts;
	float* weights;
	float score;
	int i;
	int cur_node = 0;
	const tree_node_RC* n;
	while (true) {
		n = nodes_RC_[cur_node];
		if (n->status == TERMINAL) {
			for (int i = 0; i < C_; i++)
				out[i] = (n->distribution)[i];
			break;
		} else {
			atts = n->atts;
			weights = n->weights;
			score = 0;
			for (i = 0; i < K_; i++) {
				score += point[atts[i]] * weights[i];
			}
			if (score <= n->bias) {
				cur_node = n->left;
			} else {
				cur_node = n->right;
			}
		}
	}
}

float Tree::oob_accuracy() const {
	int correct = 0;
	int total = 0;
	// Loop through training set looking for instances with weight 0
	float* temp = new float[num_attributes_];
	for (int i = 0; i < num_instances_; ++i) {
		if ((*weight_list_)[i] == 0) {
			for (int j = 0; j < num_attributes_; j++) {
				temp[j] = data_[i + j * num_instances_];
			}
			if (predict(temp) == labels_[i])
				correct++;
			total++;
		}
	}
	delete[] temp;
	return float(correct) / total;
}

float Tree::training_accuracy() const {
	float* temp = new float[num_attributes_];
	int correct = 0;
	for (int i = 0; i < num_instances_; ++i) {
		for (int j = 0; j < num_attributes_; j++) {
			temp[j] = data_[i + j * num_instances_];
		}
		if (predict(temp) == labels_[i])
			correct++;
	}
	delete[] temp;
	return float(correct) / num_instances_;
}

//fill the provided matrix (of length num_attributes_) with the total gain from
//each variable
void Tree::variable_gain(float* vars) {
	assert(splitfun_ == SINGLE); //NYI for RC

	vector<tree_node*>::const_iterator it;
	for (int i = 0; i < num_attributes_; i++)
		vars[i] = 0;
	for (it = nodes_.begin(); it != nodes_.end(); it++) {
		if ((*it)->status == SPLIT)
			vars[(*it)->attr] += (*it)->gain;
	}

}

/*
 * Find nearest neighbors of an input point (unsupervised trees only)
 * point = num_atts_-length vector containing the point whose neighbors we are searching for
 * neighbors = empty vector that will contain the identified nearest neighbors
 * dists = empty vector that will contain the mean hierarchy distances to each identified neighbor
 * num_neighbors = the number of neighbors to retrieve (equal to the length of previous 2 vectors)
 */
void Tree::nearest(const float* const point, int* const neighbors,
		float* const dists, const int num_neighbors) {
	int found = 0;
	bool done = false;
	bool foundnext;
	set<unsigned int> visited;
	unordered_set<unsigned int>::const_iterator it;
	//handle single-split case
	if (splitfun_ == SINGLE) {
		const tree_node* n;
		//base case
		unsigned int cur_node = 0;
		while (true) {
			n = nodes_[cur_node];
			if (n->status == TERMINAL) {
				//get neighbors, starting in terminal node and working up as needed
				float size = 0;
				while (true) {
					for (it = n->set1->begin(); it != n->set1->end(); it++) {
						if (found < num_neighbors) {
							neighbors[found] = *it;
							dists[found] = size;
							found++;
						} else {
							done = true;
							break;
						}
					}
					if (done)
						break;
					else {
						foundnext = false;
						while (!foundnext) {
							//go to sibling node (if not visited) and continue
							visited.insert(cur_node);
							cur_node = n->parent;
							//if we have already returned every element in the tree, just fill remainder with -1
							if (cur_node > nodes_.size()) {
								for (; found < num_neighbors; found++) {
									neighbors[found] = -1;
									dists[found] = -1.0;
								}
							}
							n = nodes_[cur_node];
							size = float(n->set1->size()) / unique_insts_;
							if (visited.count(n->left) == 0) {
								cur_node = n->left;
								n = nodes_[cur_node];
								foundnext = true;
							} else if (visited.count(n->right) == 0) {
								cur_node = n->right;
								n = nodes_[cur_node];
								foundnext = true;
							}
							//if sibling was visited already, go back to start of this loop and climb another level
						}
					}
				}
				break;
			} else {
				if (point[n->attr] <= n->split_point) {
					cur_node = n->left;
				} else {
					cur_node = n->right;
				}
			}
		}
	}
	//handle random-combination split case
	else {
		//base case
		uint32* atts;
		float* weights;
		float score;
		int i;
		unsigned int cur_node = 0;
		const tree_node_RC* n;
		while (true) {
			n = nodes_RC_[cur_node];
			if (n->status == TERMINAL) {
				//get neighbors, starting in terminal node and working up as needed
				float size = 0;
				while (true) {
					for (it = n->set1->begin(); it != n->set1->end(); it++) {
						if (found < num_neighbors) {
							neighbors[found] = *it;
							dists[found] = size;
							found++;
						} else {
							done = true;
							break;
						}
					}
					if (done)
						break;
					else {
						foundnext = false;
						while (!foundnext) {
							//go to sibling node (if not visited) and continue
							visited.insert(cur_node);
							cur_node = n->parent;
							assert(cur_node < nodes_RC_.size()); //make sure we aren't somehow at root
							n = nodes_RC_[cur_node];
							size = float(n->set1->size()) / unique_insts_;
							if (visited.count(n->left) == 0) {
								cur_node = n->left;
								n = nodes_RC_[cur_node];
								foundnext = true;
							} else if (visited.count(n->right) == 0) {
								cur_node = n->right;
								n = nodes_RC_[cur_node];
								foundnext = true;
							}
							//if sibling was visited already, go back to start of this loop and climb another level
						}
					}
				}
				break;
			} else {
				atts = n->atts;
				weights = n->weights;
				score = 0;
				for (i = 0; i < K_; i++) {
					score += point[atts[i]] * weights[i];
				}
				if (score <= n->bias) {
					cur_node = n->left;
				} else {
					cur_node = n->right;
				}
			}
		}
	}
}

/*
 * Uses an unsupervised/semi-supervised tree as a metric to compute the (dis)similarity
 * between two input data points
 *
 * point1 and point2 are both float vectors of length num_attributes_
 */
float Tree::metric_distance(const float* point1, const float* point2) {
	int cur_node = 0;
	int numnodes = nodes_RC_.size();
	float sizescore, leafscore, alpha;
	bool temp;
	if (splitfun_ == SINGLE) {
		//base case
		const tree_node* n;
		while (true) {
			n = nodes_[cur_node];
			if (n->status == TERMINAL) {
				return 0;
			} else {
				temp = point1[n->attr] <= n->split_point;
				if (temp == (point2[n->attr] <= n->split_point)) {
					if (temp)
						cur_node = n->left;
					else
						cur_node = n->right;
				} else
					return float(n->set1->size()) / unique_insts_;
			}
		}
	} else {
		//base case
		uint32* atts;
		float* weights;
		float score, score2;
		int i;
		const tree_node_RC* n;
		while (true) {
			n = nodes_RC_[cur_node];
			if (n->status == TERMINAL) {
				leafscore = leafsims_[cur_node + cur_node * numnodes];
				alpha = leafcerts_[cur_node + cur_node * numnodes];
				return alpha * leafscore + (1 - alpha) * 0;;
			} else {
				//compute scores for both points
				atts = n->atts;
				weights = n->weights;
				score = 0;
				for (i = 0; i < K_; i++) {
					score += point1[atts[i]] * weights[i];
				}
				score2 = score;
				score = 0;
				for (i = 0; i < K_; i++) {
					score += point2[atts[i]] * weights[i];
				}
				//continue until we reach the smallest node shared by both points
				temp = score <= n->bias;
				if (temp == (score2 <= n->bias)) {
					if (temp) {
						cur_node = n->left;
					} else {
						cur_node = n->right;
					}
				} else {
					sizescore = float(n->set1->size()) / unique_insts_;
					break;
				}
			}
		}
		//if we get here, then we've reached the smallest common node and it isn't a leaf node
		//now continue by finding the leaf node for each point.
		int commonl = n->left;
		int commonr = n->right;
		int leaf1, leaf2;
		//first point 1
		if (temp)
			cur_node = commonr;
		else
			cur_node = commonl;
		while (true) {
			n = nodes_RC_[cur_node];
			if (n->status == TERMINAL) {
				leaf1 = cur_node;
				break;
			} else {
				atts = n->atts;
				weights = n->weights;
				score = 0.0f;
				for (i = 0; i < K_; i++) {
					score += point1[atts[i]] * weights[i];
				}
				if (score <= n->bias)
					cur_node = n->left;
				else
					cur_node = n->right;
			}
		}
		//now point 2
		if (temp)
			cur_node = commonl;
		else
			cur_node = commonr;
		while (true) {
			n = nodes_RC_[cur_node];
			if (n->status == TERMINAL) {
				leaf2 = cur_node;
				break;
			} else {
				atts = n->atts;
				weights = n->weights;
				score = 0.0f;
				for (i = 0; i < K_; i++) {
					score += point2[atts[i]] * weights[i];
				}
				if (score <= n->bias)
					cur_node = n->left;
				else
					cur_node = n->right;
			}
		}
		//now use both leaf nodes to look up leaf similarity and certainty for the pair
		leafscore = leafsims_[leaf1 + leaf2 * numnodes];
		alpha = leafcerts_[leaf1 + leaf2 * numnodes];
		//use the certainty score to combine the leafscore and sizescore, and return the result
		return alpha * leafscore + (1 - alpha) * sizescore;
	}
}

/*
 * Uses an unsupervised/semi-supervised tree as a metric to compute the (dis)similarity
 * between an input data point and another point that is already a member of the forest.
 *
 * p1 = a float vector of length num_attributes_
 * p2 = an integer indicating the id of an element in the forest
 */
float Tree::metric_distance(const float* point1, const int point2) {
	assert(point2 < num_instances_);
	int cur_node = 0;
	float sizescore, leafscore, alpha;
	bool temp, temp2;
	if (splitfun_ == SINGLE) {
		//NOT YET IMPLEMENTED
		assert(0 == 1);
	} else {
		uint32* atts;
		float* weights;
		float score, score2;
		int i;
		const tree_node_RC* n;
		const tree_node_RC* tempn;
		while (true) {
			n = nodes_RC_[cur_node];
			if (n->status == TERMINAL) {
				return 0;
			} else {
				//compute score for point1
				atts = n->atts;
				weights = n->weights;
				score = 0;
				for (i = 0; i < K_; i++) {
					score += point1[atts[i]] * weights[i];
				}
				//check child nodes to see which one point2 is in
				tempn = nodes_RC_[n->left];
				temp2 = tempn->set1->find(point2) != tempn->set1->end();
				//continue until we reach the smallest node shared by both points
				temp = score <= n->bias;
				if (temp == temp2) {
					if (temp) {
						cur_node = n->left;
					} else {
						cur_node = n->right;
					}
				} else {
					sizescore = float(n->set1->size()) / unique_insts_;
					break;
				}
			}
		}
		//if we get here, then we've reached the smallest common node and it isn't a leaf node
		//now continue by finding the leaf node for each point.
		int commonl = n->left;
		int commonr = n->right;
		int leaf1, leaf2;
		int numnodes = nodes_RC_.size();
		//first point 1
		if (temp)
			cur_node = commonl;
		else
			cur_node = commonr;
		while (true) {
			n = nodes_RC_[cur_node];
			if (n->status == TERMINAL) {
				leaf1 = cur_node;
				break;
			} else {
				atts = n->atts;
				weights = n->weights;
				score = 0;
				for (i = 0; i < K_; i++) {
					score += point1[atts[i]] * weights[i];
				}
				if (score <= n->bias)
					cur_node = n->left;
				else
					cur_node = n->right;
			}
		}
		//now point 2
		if (temp)
			cur_node = commonr;
		else
			cur_node = commonl;
		while (true) {
			n = nodes_RC_[cur_node];
			if (n->status == TERMINAL) {
				leaf2 = cur_node;
				break;
			} else {
				tempn = nodes_RC_[n->left];
				temp2 = tempn->set1->find(point2) != tempn->set1->end();
				if (temp2)
					cur_node = n->left;
				else
					cur_node = n->right;
			}
		}
		//now use both leaf nodes to look up leaf similarity and certainty for the pair
		leafscore = leafsims_[leaf1 + leaf2 * numnodes];
		alpha = leafcerts_[leaf1 + leaf2 * numnodes];
		//use the certainty score to combine the leafscore and sizescore, and return the result
		return alpha * leafscore + (1 - alpha) * sizescore;
	}
}

//generate scores for all variables
//void Tree::variable_importance(vector<float>* score, uint32* seed) const {
//	// build subset
//	InstanceSet* subset = InstanceSet::create_subset(set_, *weight_list_);
//	// get the oob accuracy before we start
//	int correct = 0;
//	for (int i = 0; i < subset->size(); ++i) {
//		if (predict(*subset, i) == subset->label(i)) {
//			correct++;
//		}
//	}
//	float oob_acc = oob_accuracy();
//	score->resize(set_.num_attributes());
//	for (int i = 0; i < set_.num_attributes(); ++i) {
//		if (vars_used_.find(i) != vars_used_.end()) {
//			// make a backup copy of the variable
//			vector<float> backup;
//			subset->save_var(i, &backup);
//			// shuffle the values in this variable around
//			subset->permute(i, seed);
//			int permuted = 0;
//			for (int j = 0; j < subset->size(); ++j) {
//				if (predict(*subset, j) == subset->label(j)) {
//					permuted++;
//				}
//			}
//			// decrease in accuracy!
//			(*score)[i] = (correct - permuted);
//			// restore the proper stuff
//			subset->load_var(i, backup);
//		} else {
//			(*score)[i] = 0.0;
//		}
//	}
//	delete subset;
//}
//bool Tree::oob(int instance_no) const {
//	return ((*weight_list_)[instance_no] == 0);
//}

