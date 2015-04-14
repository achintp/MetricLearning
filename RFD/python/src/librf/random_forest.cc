/**
 * @file
 * @brief random forest implementation
 */
#include "random_forest.h"
#include <assert.h>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <functional>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

/*
 * Computes the closure of a set of must-link and cannot-link constraints (i.e. finds all
 * sets of points linked together by chains of pairwise constraints, and then identifies
 * cannot-link constraints between these sets (where two sets A and B are negatively linked if
 * there exists a cannot-link constraint from any point in A to any point in B).
 *
 * pairs = a numpairs by 3integer array, where each row describes a single pair-constraint
 * 	by listing the ids of the two constrained points, followed by a +1 or -1 (indicating,
 * 	respectively,a must-link or cannot-link constraint).
 * numpairs = the number of constraints described in pairs (i.e. the length of pairs divided by 3)
 *
 * returns: a pointer to a closure object, describing the set of constrained point sets and the
 * 	cannot-link constraints between them
 */
topclosure* ClosureUtils::pairwise_closure(const int* pairs,
		const int numpairs) {
	topclosure* out = new topclosure();
	int matchid1, matchid2;
	unordered_map<int, int>::const_iterator it;
	unordered_set<int>* temp;
	unordered_set<int>* temp2;
	unordered_set<int>::const_iterator it2;
	int nextid = 0;

	//iterate through the list of pair constraints, for now only handling must-link
	for (const int* i = pairs; i < pairs + numpairs * 3; i += 3) {
		if (*(i + 2) > 0) {
			//first determine the current set membership (if any) of each point in the constraint
			it = out->setmembership.find(*i);
			if (it == out->setmembership.end())
				matchid1 = -1;
			else
				matchid1 = it->second;
			it = out->setmembership.find(*(i + 1));
			if (it == out->setmembership.end())
				matchid2 = -1;
			else
				matchid2 = it->second;

			//now take action based on the identified set membership:
			if (matchid1 == matchid2) {
				//if both points have no set membership, create a new set containing them
				if (matchid1 == -1) {
					out->setmembership[*i] = nextid;
					out->setmembership[*(i + 1)] = nextid;
					temp = &(out->pointsets[nextid]);
					temp->insert(*i);
					temp->insert(*(i + 1));
					nextid++;
				}
				//if both points are in the same set already, do nothing
			} else {
				//if ONE of the points has no set membership, add it to the other point's set
				if (matchid1 == -1) {
					out->setmembership[*i] = matchid2;
					out->pointsets[matchid2].insert(*i);
				} else if (matchid2 == -1) {
					out->setmembership[*(i + 1)] = matchid1;
					out->pointsets[matchid1].insert(*(i + 1));
				}
				//and if both points have different set memberships, merge set 2 into set 1
				else {
					temp = &(out->pointsets[matchid1]);
					temp2 = &(out->pointsets[matchid2]);
					for (it2 = temp2->begin(); it2 != temp2->end(); it2++) {
						out->setmembership[*it2] = matchid1;
						temp->insert(*it2);
					}
					out->pointsets.erase(matchid2);
				}
			}
		}
	}
	unordered_set<pair<int, int>, pair_hash, pair_eq<int> > tempneglinks;
	unordered_set<pair<int, int>, pair_hash, pair_eq<int> >::const_iterator neglinksit;

	//now iterate through the list of constraints again, this time handling the negative constraints
	for (const int* i = pairs; i < pairs + numpairs * 3; i += 3) {
		if (*(i + 2) < 0) {
			//again start by determining the current set membership (if any) of each point in the constraint
			//if either point has no set membership, create a new set containing only that point
			it = out->setmembership.find(*i);
			if (it == out->setmembership.end()) {
				out->setmembership[*i] = nextid;
				out->pointsets[nextid].insert(*i);
				matchid1 = nextid;
				nextid++;
			} else
				matchid1 = it->second;
			it = out->setmembership.find(*(i + 1));
			if (it == out->setmembership.end()) {
				out->setmembership[*(i + 1)] = nextid;
				out->pointsets[nextid].insert(*(i + 1));
				matchid2 = nextid;
				nextid++;
			} else
				matchid2 = it->second;

			//to ensure uniqueness, make sure all pairs are ordered such that the lower set id comes first
			if (matchid2 < matchid1)
				swap(matchid1, matchid2);

			//now create a pair out of the two set ids (whether old or new) denoting the cannot-link constraint
			out->neglinkset.insert(make_pair(matchid1, matchid2));
			//the fact that neglinks is a set will ensure no duplicate pairs
		}
	}
	//now that the set structure has ensured they are unique, place the negative links in a vector
	out->neglinks.resize(out->neglinkset.size());
	int j = 0;
	for (neglinksit = out->neglinkset.begin();
			neglinksit != out->neglinkset.end(); neglinksit++) {
		out->neglinks[j] = *neglinksit;
		j++;
	}

	return out;
}

/*
 * Transforms a given constraint closure object into an actual complete set of explicit pairwise
 * constraints.
 */
constraint_set* ClosureUtils::constraints_from_closure(const closure* close) {
	vector<int>* cons = new vector<int>;

	//first extract must-link constraints
	unordered_map<int, unordered_set<int> >::const_iterator it;
	unordered_set<int>::const_iterator it2, it3;
	unordered_set<int> curset1, curset2;
	for (it = close->pointsets.begin(); it != close->pointsets.end(); it++) {
		for (it2 = it->second.begin(); it2 != it->second.end(); it2++) {
			it3 = it2;
			it3++;
			for (; it3 != it->second.end(); it3++) {
				cons->push_back(*it2);
				cons->push_back(*it3);
				cons->push_back(1);
			}
		}
	}

	//then get cannot-link constraints
	for (int i = 0; i < close->neglinks.size(); i++) {
		curset1 = close->pointsets.at(close->neglinks.at(i).first);
		curset2 = close->pointsets.at(close->neglinks.at(i).second);
		for (it2 = curset1.begin(); it2 != curset1.end(); it2++) {
			for (it3 = curset2.begin(); it3 != curset2.end(); it3++) {
				cons->push_back(*it2);
				cons->push_back(*it3);
				cons->push_back(-1);
			}
		}
	}
	constraint_set* out = new constraint_set(cons);
	return out;
}

RandomForest::RandomForest() {
}

/**
 * @param data 2D float array containing data (ordered as attributes by instances)
 * @param labels 1D int array containing labels
 * @param atts the number of attributes
 * @param insts the number of instances
 * @param num_trees #trees to train
 * @param K #random vars to consider at each split
 * @param min_size minimum number of objects allowed in a leaf node
 * @param threads the number of threads to use for training/testing forests in parallel
 */
RandomForest::RandomForest(const float *data, int atts, int insts,
		const int *labels, int insts2, int num_trees, int K, int F,
		int min_size, int splitfun, int threads, bool compute_unique) :
		K_(min(K, atts)), F_(F), min_size_(min_size), threads_(threads), num_atts_(
				atts), splitfun_(splitfun), super_(true) {
	assert(insts2 == insts);
	assert(splitfun_ == SINGLE || splitfun == RC);

	//if specified, compute unique values in each dimension
	int i, j;
	if (!compute_unique || splitfun == RC)
		unique_vals_ = NULL;
	else {
		vector < unordered_set<float> > temp;
		temp.resize(atts);
		for (i = 0; i < atts; i++) {
			for (j = 0; j < insts; j++)
				temp[i].insert(data[i + j * atts]);
		}

		unique_vals_ = new vector<vector<float> >;
		unique_vals_->resize(atts);
		unordered_set<float>::const_iterator it;
		for (i = 0; i < atts; i++)
			for (it = temp[i].begin(); it != temp[i].end(); it++)
				(*unique_vals_)[i].push_back(*it);
	}

	//determine number of classes
	C_ = 1;
	for (i = 0; i < insts; i++) {
		if (labels[i] >= C_)
			C_ = labels[i] + 1;
	}
	assert(C_ >= 2);

	//now generate trees
	//class_weights_.resize(2, 1);
	mom::Semaphore* sem = new mom::Semaphore(0, threads_);
	boost::thread_group tg;
	for (i = 0; i < num_trees; ++i) {
		weight_list* w = new weight_list(insts, insts);
		//sample with replacement
		for (j = 0; j < insts; ++j) {
			if (w->add(rand() % insts) == -1)
				j--;
		}
		Tree* tree;
		tree = new Tree(data, atts, insts, labels, w, K_, F_, rand(), min_size_,
				0, splitfun_, C_, unique_vals_);
		//wait for a thread slot to become available
		sem->WaitOne();
		//create thread for training
		boost::thread* tp = new boost::thread(traintree, tree, i, sem);
		tg.add_thread(tp);
		trees_.push_back(tree);
	}
	tg.join_all();
	delete sem;
}

/**
 * @param data 2D float array containing data (ordered as attributes by instances)
 * @param constraints num_cons by 3 2D int array describing num_cons dual triplet or pairwise constraints on the data
 * @param atts the number of attributes
 * @param insts the number of instances
 * @param num_cons the number of triplet constraints
 * @param num_trees #trees to train
 * @param K #random vars to consider at each split
 * @param min_size minimum number of objects allowed in a leaf node
 * @param threads the number of threads to use for training/testing forests in parallel
 */
RandomForest::RandomForest(const float *data, int atts, int insts,
		const int* constraints, int num_cons, int three, int num_trees, int K,
		int F, int min_size, int splitfun, int threads, bool compute_unique,
		float splitweight, float distweight, float satweight, float certfactor) :
		K_(min(K, atts)), F_(F), min_size_(min_size), threads_(threads), num_atts_(
				atts), splitfun_(splitfun), super_(false) {
	assert(splitfun_ == SINGLE || splitfun == RC);
	assert(three == 3);

	//if specified, compute unique values in each dimension
	int i, j;
	if (!compute_unique || splitfun == RC)
		unique_vals_ = NULL;
	else {
		vector < unordered_set<float> > temp;
		temp.resize(atts);
		for (i = 0; i < atts; i++) {
			for (j = 0; j < insts; j++)
				temp[i].insert(data[i + j * atts]);
		}

		unique_vals_ = new vector<vector<float> >;
		unique_vals_->resize(atts);
		unordered_set<float>::const_iterator it;
		for (i = 0; i < atts; i++)
			for (it = temp[i].begin(); it != temp[i].end(); it++)
				(*unique_vals_)[i].push_back(*it);
	}

	//determine whether we are using triplet or pairwise constraints (or no constraints)
	ConstraintType contype;
	if (num_cons == 0)
		contype = NONE;
	else
		contype = TRIPLET;
	for (const int* it = constraints + 2; it < constraints + num_cons * 3; it +=
			3) {
		if (*it < 0) {
			contype = PAIR;
			break;
		}
	}

	//if using pairwise constraints, compute a constraint closure over the constraints
	topclosure* masterclosure = NULL;
	if (contype == PAIR)
		masterclosure = ClosureUtils::pairwise_closure(constraints, num_cons);

	//now generate trees
	//class_weights_.resize(2, 1);
	mom::Semaphore* sem = new mom::Semaphore(0, threads_);
	boost::thread_group tg;
	for (i = 0; i < num_trees; i++) {
		//generate weight list - for semi-supervised forests, ensure that all points have weight of at least 1
		weight_list* w = new weight_list(insts, insts, 1);
		//now add <insts> total weight to the points, distributed randomly
		for (j = 0; j < insts; j++) {
			if (w->add(rand() % insts) == -1)
				j--;
		}
		//keep track of how many times each point appears (obsolete, but needed elsewhere for the moment)
		for (j = 0; j < insts; j++) {
			if ((*w)[j] > 0) {
				if (appear_counts_.find(j) == appear_counts_.end())
					appear_counts_.insert(make_pair(j, 1));
				else
					appear_counts_[j]++;
			}
		}
		Tree* tree;
		tree = new Tree(data, atts, insts, constraints, num_cons, w, K_, F_,
				rand(), contype, min_size_, 0, splitfun_, unique_vals_,
				masterclosure, splitweight, distweight, satweight, certfactor);
		//wait for a thread slot to become available
		sem->WaitOne();
		//create thread for training
		boost::thread* tp = new boost::thread(traintree, tree, i, sem);
		tg.add_thread(tp);
		trees_.push_back(tree);
	}
	tg.join_all();
	delete sem;
	delete masterclosure;
}

RandomForest::~RandomForest() {
	for (int i = 0; i < trees_.size(); ++i) {
		delete trees_[i];
	}
	if (unique_vals_ != NULL)
		delete unique_vals_;
}

void RandomForest::traintree(Tree* reftotrain, int treenum,
		mom::Semaphore* sem) {
	cout << "Starting Thread: " << treenum << endl;
	reftotrain->grow();
	cout << "Grew tree " << treenum << endl;
	sem->Release();
}

int RandomForest::ntrees() {
	return trees_.size();
}

void RandomForest::write(const char filename[]) {
	ofstream out;
	out.open(filename);
	write(out);
	out.close();
}

void RandomForest::write(ostream& o) {
	o << trees_.size() << " " << K_ << " " << num_atts_ << " " << splitfun_
			<< " " << C_ << endl;
	for (int i = 0; i < trees_.size(); ++i) {
		trees_[i]->write(o);
	}
}

void RandomForest::read(const char filename[]) {
	ifstream in;
	in.open(filename);
	read(in);
	in.close();
}

void RandomForest::read(istream& in) {
	unique_vals_ = NULL;
	int num_trees;
	super_ = true;
	in >> num_trees >> K_ >> num_atts_ >> splitfun_ >> C_;
	for (int i = 0; i < num_trees; ++i) {
		trees_.push_back(new Tree(in, K_, splitfun_));
	}
}

/*
 * Classify each of a set of unlabelled inputs and return a 2D boolean array of tree votes
 * @param data instances by attributes float array containing the dataset to be labelled
 * @param atts the number of attributes
 * @param insts the number of instances
 * @param votes instances by num_trees byte array to hold the votes from each tree for each instance
 */
void RandomForest::votesset(const float *data, int insts, int atts,
		signed char *outvotes) {
	int ntrees = trees_.size();
	boost::thread_group tg;
	int morethreads = threads_ * 2;
	mom::Semaphore* sem = new mom::Semaphore(0, morethreads);
	for (int i = 0; i < ntrees; ++i) {
		//wait for a thread slot to become available
		sem->WaitOne();
		//create thread for testing
		Tree* tree = trees_[i];
		boost::thread* tp = new boost::thread(gettreevotes, tree, data,
				outvotes, atts, insts, i, ntrees, sem, splitfun_);
		tg.add_thread(tp);
	}
	tg.join_all();
	delete sem;
}

/*
 * Parallel helper method for filling in each tree's responses in parallel
 */
void RandomForest::gettreevotes(Tree* tree, const float* data,
		signed char* outvotes, int atts, int insts, int treenum, int ntrees,
		mom::Semaphore* sem, int splitfun) {
	outvotes = outvotes + treenum;
	if (splitfun == SINGLE) {
		for (int i = 0; i < insts; i++) {
			*outvotes = tree->predict(data);
			outvotes = outvotes + ntrees;
			data = data + atts;
		}
	} else {
		for (int i = 0; i < insts; i++) {
			*outvotes = tree->predict_RC(data);
			outvotes = outvotes + ntrees;
			data = data + atts;
		}
	}
	sem->Release();
}

/*
 * Perform regression on the input dataset, returning a regression score for each class
 * for every instance
 * @param data instances by attributes float array containing the dataset to be labelled
 * @param atts the number of attributes
 * @param insts the number of instances
 * @param regs instances-by classes matrix that will contain per-class regression results
 * 	(all values should be initialized to 0)
 * @param insts2 length of regs, for easy numpy linkage via swig
 * @param C number of classes in the forest
 */
void RandomForest::regrset(const float *data, int insts, int atts, float *regs,
		int insts2, int C) {
	assert(insts == insts2);
	assert(C_ == C);
	int ntrees = trees_.size();
	boost::thread_group tg;
	int morethreads = threads_;
	mom::Semaphore* sem = new mom::Semaphore(0, morethreads);
	boost::mutex* lock = new boost::mutex();
	for (int i = 0; i < ntrees; i++) {
		//wait for a thread slot to become available
		sem->WaitOne();
		//create thread for testing
		Tree* tree = trees_[i];
		boost::thread* tp = new boost::thread(gettreeregs, tree, data, regs,
				atts, insts, ntrees, sem, lock, splitfun_);
		tg.add_thread(tp);
	}
	tg.join_all();

	delete sem;
	delete lock;
}

/*
 * Parallel helper method for filling in each tree's responses in parallel
 */
void RandomForest::gettreeregs(Tree* tree, const float* data, float* regs,
		int atts, int insts, int ntrees, mom::Semaphore* sem,
		boost::mutex* lock, int splitfun) {
//iterate through data and apply this tree to each instance, saving
//results in temporary array
	int C = tree->C_;
	float* tempregs = new float[insts * C];
	float s;
	if (splitfun == SINGLE) {
		for (int i = 0; i < insts; i++)
			tree->regress(data + i * atts, tempregs + i * C, s);
	} else {
		for (int i = 0; i < insts; i++) {
			tree->regress_RC(data + i * atts, tempregs + i * C);
		}
	}
//obtain lock, then add temporary array results to output array
	lock->lock();
	for (int i = 0; i < insts * C; i++)
		regs[i] += tempregs[i];
	lock->unlock();
	delete[] tempregs;
	sem->Release();
}

/*
 * For use with RFD: performs regression on point pairs generated from two data matrices
 * to build an RFD kernel matrix between the two.
 * @param data instances by attributes float array containing the first dataset
 * @param atts the number of attributes
 * @param insts the number of instances in the first dataset
 * @param data2 instances by attributes float array containing the second dataset
 * @param atts2 equal to atts
 * @param insts2 the number of instances in the second dataset
 * @param dists insts by insts2 array that will hold the generated kernel matrix
 * @param insts12 equal to insts
 * @param insts22 equal to insts2
 * @param position if true, then absolute position is used in the pair vectors
 */
void RandomForest::rfdregr(const float* data, int insts, int atts,
		const float* data2, int insts2, int atts2, float* dists, int insts12,
		int insts22, bool position) {
	assert(atts == atts2);
	assert(insts12 == insts);
	assert(insts22 == insts2);
	assert(C_ == 2);
	//
	// float* codes = new float[insts*insts2];
	//
	if (position)
		assert(atts * 2 == num_atts_);
	else
		assert(atts == num_atts_);
	cout << "Now computing RFD distance matrix: " << insts << " by " << insts2
			<< " instances." << endl;

	boost::thread_group tg;
	mom::Semaphore* sem = new mom::Semaphore(0, threads_);
	for (int i = 0; i < insts; i++) {
		//wait for a thread slot to become available
		sem->WaitOne();
		//create thread for testing
		boost::thread * tp = new boost::thread(rfdregrhelper, trees_,
				data + i * atts, data2, insts2, atts, dists + i * insts2, sem,
				splitfun_, position);
		tg.add_thread(tp);
	}
//make sure all threads are finished
	tg.join_all();
	delete sem;
	for(int i = 0; i < insts; i++){
		cout << "[";
		for(int j = 0; j < insts2; j++){
			cout << dists[j] << endl;
		}
		cout <<  "]" << endl << endl;
	}
}


void RandomForest::rfdcode(const float* data, int insts, int atts, float* codes,
	int insts1, int trsize, bool position) {
	assert(C_ == 2);
	assert(trsize == ntrees());
	if(position)
		assert(false);
	cout << "Now computing RFD codes: " << insts << "by " << trsize << endl;

	boost::thread_group tg;
	mom::Semaphore* sem = new mom::Semaphore(0, threads_);
	for(int i = 0; i < insts; i++){
		sem->WaitOne();
		boost::thread *tp = new boost::thread(rfdcodehelper, trees_, data + i*atts,
			atts, codes + i*trsize, sem);
		tg.add_thread(tp);
	}

	tg.join_all();
	delete sem;

	for(int i = 0; i < insts; i++){
		cout << "[";
		for(int j = 0; j < trsize; j++){
			cout << codes[i*trsize + j];
		}
		cout <<  "]" << endl << endl;
	}
}
/*
 Parallel helper method for filling in code words from each tree
*/
void RandomForest::rfdcodehelper(vector<Tree*>& trees, const float* point, 
	int atts, float* codes, mom::Semaphore* sem){
	int ntrees = trees.size();

	float* tdat = new float[atts];
	float *temp = new float[2];

	for(int j = 0; j < atts; j++){
		tdat[j] = point[j];
	}

	for(int j = 0; j < ntrees; j++){
		codes[j] = 0;
		float s;
		trees[j]->regress(tdat, temp, s);
		codes[j] = s;
	}

	delete[] tdat;
	delete[] temp;
	sem->Release();
}

/*
 * Parallel helper method for filling in each tree's responses in parallel
 */
void RandomForest::rfdregrhelper(vector<Tree*>& trees, const float* point,
		const float* data2, int insts, int atts, float* dists,
		mom::Semaphore* sem, int splitfun, bool position) {
	int ntrees = trees.size();
	int i, j;
	float* tdat;
	if (position)
		tdat = new float[atts * 2];
	else
		tdat = new float[atts];
	const float* point2;
	float* temp = new float[2];
	//
	// float *codes = new float[insts];
	//
	if (splitfun == SINGLE) {
		if (position) {
			for (i = 0; i < insts; i++) {
				//first build pair data vector
				point2 = data2 + i * atts;
				for (j = 0; j < atts; j++)
					tdat[j] = abs(point[j] - point2[j]);

				for (j = 0; j < atts; j++)
					tdat[j + atts] = (point[j] + point2[j]) / 2;
				//now sum tree regression responses to it
				dists[i] = 0;
				for (j = 0; j < ntrees; j++) {
					float s;
					trees[j]->regress(tdat, temp, s);
					dists[i] += temp[1];
					dists[i] = s;
				}
			}
		} else {
			for (i = 0; i < 1; i++) {
				//first build pair data vector
				point2 = data2 + i * atts;
				for (j = 0; j < atts; j++)
					// tdat[j] = abs(point[j] - point2[j]);
					tdat[j] = point[j];
				//now sum tree regression responses to it
				dists[i] = 0;
				for (j = 0; j < ntrees; j++) {
					float s;
					trees[j]->regress(tdat, temp, s);
					dists[i] += temp[1];
					dists[i] = s;
				}
			}
		}
	} else {
		if (position) {
			for (i = 0; i < insts; i++) {
				//first build pair data vector
				point2 = data2 + i * atts;
				for (j = 0; j < atts; j++)
					tdat[j] = abs(point[j] - point2[j]);

				for (j = 0; j < atts; j++)
					tdat[j + atts] = (point[j] + point2[j]) / 2;
				//now sum tree regression responses to it
				dists[i] = 0;
				for (j = 0; j < ntrees; j++) {
					trees[j]->regress_RC(tdat, temp);
					dists[i] += temp[1];
				}
			}
		} else {
			for (i = 0; i < insts; i++) {
				//first build pair data vector
				point2 = data2 + i * atts;
				for (j = 0; j < atts; j++)
					tdat[j] = abs(point[j] - point2[j]);
				//now sum tree regression responses to it
				dists[i] = 0;
				for (j = 0; j < ntrees; j++) {
					trees[j]->regress_RC(tdat, temp);
					dists[i] += temp[1];
				}
			}
		}
	}
	// cout << "[";
	// for(int i = 0; i < insts; i++){
	// 	cout << codes[i] << endl;
	// }
	// cout <<  "]" << endl << endl;
	delete[] temp;
	delete[] tdat;
	sem->Release();
}

//compute mean variable importance for each attribute
void RandomForest::variable_importance(float* vars, int atts) {
	assert(atts == num_atts_);

	for (int i = 0; i < atts; i++)
		vars[i] = 0;

	float* temp = new float[atts];
	for (int i = 0; i < trees_.size(); i++) {
		trees_[i]->variable_gain(temp);
		for (int i = 0; i < atts; i++)
			vars[i] += temp[i];
	}

	for (int i = 0; i < atts; i++) {
		vars[i] /= trees_.size();
	}
}

/*
 * Find nearest neighbors of multiple input points (unsupervised forests only)
 * point = m by num_atts_ vector containing the points whose neighbors we are searching for
 * 	(where m is the number of points being queried)
 * neighbors = empty m by num_neighbors array that will contain the identified nearest neighbors
 * dists = empty m by num_neighbors array that will contain the mean hierarchy distances to each identified neighbor
 * num_neighbors = the number of neighbors to retrieve
 */
void RandomForest::nearest_multi(const float* points, const int num_points,
		int* neighbors, float* dists, const int num_neighbors,
		const int num_candidates) {
	assert(!super_);
	//first divy input points up between the different threads
	int i;
	vector < vector<int> > workerpts;
	workerpts.resize(threads_);
	for (i = 0; i < num_points; i++)
		workerpts[i % threads_].push_back(i);
	//now dispatch each set of points to a different worker in a different thread
	boost::thread_group tg;
	for (i = 0; i < threads_; i++) {
		boost::thread* tp = new boost::thread(nearest_multi_worker, points,
				neighbors, dists, num_neighbors, num_candidates, workerpts[i],
				this);
		tg.add_thread(tp);
	}
	//make sure all threads are finished
	tg.join_all();
}

//parallel helper method for nearest neighbor retrieval
void RandomForest::nearest_multi_worker(const float* points, int* neighbors,
		float* dists, const int num_neighbors, const int num_candidates,
		const vector<int>& workerpts, const RandomForest* forest) {
	int atts = forest->num_atts_;
	vector<int>::const_iterator it;
	1 == 1;
	for (it = workerpts.begin(); it != workerpts.end(); it++) {
		forest->nearest(points + *it * atts, neighbors + *it * num_neighbors,
				dists + *it * num_neighbors, num_neighbors, num_candidates);
	}
	2 == 2;
}

/*
 * Find nearest neighbors of an input point (un/semi-supervised forests only)
 * point = num_atts_-length vector containing the point whose neighbors we are searching for
 * neighbors = empty vector that will contain the identified nearest neighbors
 * dists = empty vector that will contain the mean hierarchy distances to each identified neighbor
 * num_neighbors = the number of neighbors to retrieve (equal to the length of previous 2 vectors)
 */
void RandomForest::nearest(const float* point, int* neighbors, float* dists,
		const int num_neighbors, const int num_candidates) const {
	int i;
	const int* it;
	unordered_set<int> candidates;
	unordered_set<int>::const_iterator it2;
	int* const tneigh = new int[num_candidates];
	float* const tdists = new float[num_candidates]; //currently ignored
	float temp;
	//get neighbors and distances from each tree
	for (i = 0; i < trees_.size(); i++) {
		trees_[i]->nearest(point, tneigh, tdists, num_candidates);
		for (it = tneigh; it != tneigh + num_candidates; it++)
			candidates.insert(*it);
	}

	if (candidates.size() < num_neighbors) {
		cout << "Error, not enough nearest neighbor candidates." << endl;
		throw 1;
	}

	//compute (and sort) actual distances to each unique candidate
	vector<pair<float, int>> canddists;
	i = 0;
	for (it2 = candidates.begin(); it2 != candidates.end(); it2++) {
		if (*it2 >= 0) //ignore -1s, which represent null returns
			canddists.push_back(make_pair(metric_distance(point, *it2), *it2));
	}
	sort(canddists.begin(), canddists.end(),
			[](pair<float, int> p1, pair<float,int> p2) {return p1.first < p2.first;});
	delete[] tneigh;
	delete[] tdists;

	//fill neighbors/dists with the num_neighbors most similar items
	for (i = 0; i < num_neighbors; i++) {
		neighbors[i] = canddists[i].second;
		dists[i] = canddists[i].first;
	}
}

/*
 * Uses an unsupervised/semisupervised forest as a metric to compute the (dis)similarity
 * between two input data points
 *
 * p1 and p2 are both float vectors of length num_atts_
 */
float RandomForest::metric_distance(const float* p1, const float* p2) const {
	float out = 0;
	for (vector<Tree*>::const_iterator it = trees_.begin(); it != trees_.end();
			it++) {
		out += (*it)->metric_distance(p1, p2);
	}
	return out / trees_.size();
}

/*
 * Uses an unsupervised/semi-supervised tree as a metric to compute the (dis)similarity
 * between an input data point and another point that is already a member of the forest.
 *
 * p1 = a float vector of length num_attributes_
 * p2 = an integer indicating the id of an element in the forest
 */
float RandomForest::metric_distance(const float* p1, int p2) const {
	float out = 0;
	for (vector<Tree*>::const_iterator it = trees_.begin(); it != trees_.end();
			it++) {
		out += (*it)->metric_distance(p1, p2);
	}
	return out / trees_.size();
}

/*
 * Uses an unsupervised/semisupervised forest to compute metric distance between a
 * number of input points.
 * p1 and p2 = m by num_atts_ float arrays containing the 2 sets of m points to be compared
 * 	under the forest metric
 * out = m-length float vector that will hold the outputs
 * m = the number of point-pairs being compared
 */
void RandomForest::metric_multi(const float* p1, const float* p2, float* out,
		const int m) const {
	assert(!super_);
	//first divy input points up between the different threads
	int i;
	vector < vector<int> > workerpts;
	workerpts.resize(threads_);
	for (i = 0; i < m; i++)
		workerpts[i % threads_].push_back(i);
	//now dispatch each set of points to a different worker in a different thread
	boost::thread_group tg;
	for (i = 0; i < threads_; i++) {
		boost::thread* tp = new boost::thread(metric_multi_worker, p1, p2, out,
				m, workerpts[i], this);
		tg.add_thread(tp);
	}
	//make sure all threads are finished
	tg.join_all();
}

//Parallel helper method for metric_multi
void RandomForest::metric_multi_worker(const float* p1, const float* p2,
		float* out, const int m, const vector<int>& workerpts,
		const RandomForest* forest) {
	int atts = forest->num_atts_;
	vector<int>::const_iterator it;
	for (it = workerpts.begin(); it != workerpts.end(); it++)
		out[*it] = forest->metric_distance(p1 + *it * atts, p2 + *it * atts);
}

/*
 * 	(un/semisupervised forests only)
 *	Computes an n by m distance matrix between two sets of n and m points.  Each row of the
 *	matrix contains the distances from 1 point in the first set to all m points in the
 *	second set.
 */
void RandomForest::metric_matrix(const float* p1, const int n, const int d1,
		const float* p2, const int m, const int d2, float* out, const int n2,
		const int m2) const {
	assert(!super_);
	//first divy input points up between the different threads
	int i;
	vector < vector<int> > workerpts;
	workerpts.resize(threads_);
	for (i = 0; i < n; i++)
		workerpts[i % threads_].push_back(i);
	//now dispatch each set of points to a different worker in a different thread
	boost::thread_group tg;
	for (i = 0; i < threads_; i++) {
		boost::thread* tp = new boost::thread(metric_matrix_worker, p1, p2, out,
				n, m, workerpts[i], this);
		tg.add_thread(tp);
	}
	//make sure all threads are finished
	tg.join_all();
}

//parallel helper method for metric distance computation
void RandomForest::metric_matrix_worker(const float* p1, const float* p2,
		float* out, const int n, const int m, const vector<int>& workerpts,
		const RandomForest* forest) {
	int atts = forest->num_atts_;
	vector<int>::const_iterator it;
	int i;
	for (it = workerpts.begin(); it != workerpts.end(); it++) {
		for (i = 0; i < m; i++) {
			out[(*it * m) + i] = forest->metric_distance(p1 + *it * atts,
					p2 + i * atts);
		}
	}
	2 == 2;
}

//void RandomForest::variable_importance(vector<pair<float, int> >*ranking,
//		unsigned int* seed) const {
//	vector<float> importances(set_->num_attributes(), 0.00);
//	// Zero-out importances
//	for (int i = 0; i < trees_.size(); ++i) {
//		vector<float> tree_importance;
//		trees_[i]->variable_importance(&tree_importance, seed);
//		//aggregate
//		for (int j = 0; j < tree_importance.size(); ++j) {
//			importances[j] += tree_importance[j];
//		}
//	}
//	// Get the mean of scores
//	vector<float> raw_scores;
//	float sum = 0;
//	float sum_of_squares = 0;
//	for (int i = 0; i < importances.size(); ++i) {
//		float avg = importances[i] / trees_.size();
//		assert(avg == avg);
//		raw_scores.push_back(avg);
//		sum += avg;
//		sum_of_squares += (avg * avg);
//	}
//	float mean = sum / importances.size();
//	assert(mean == mean);
//	float std = sqrt(sum_of_squares / importances.size() - mean * mean);
//	assert(std == std);
//
//	// Write the z-scores
//	for (int i = 0; i < raw_scores.size(); ++i) {
//		float raw = raw_scores[i];
//		float zscore = 0;
//		if (std != 0) {
//			zscore = (raw - mean) / std;
//		}
//		assert(zscore == zscore);
//		ranking->push_back(make_pair(zscore, i));
//	}
//	// Sort
//	sort(ranking->begin(), ranking->end(), greater<pair<float, int> > ());
//}
