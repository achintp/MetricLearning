/* tree_node.cc
 * 
 * The only implementation here is in reading/writing
 *
 */
#include "tree_node.h"
#include <iostream>

void tree_node::write(ostream& o, int C) const {
	// we shouldn't be saving any other kind of node
	assert(status == TERMINAL || status == SPLIT);
	o << int(status);
	switch (status) {
	case TERMINAL:
		o << " " << int(label);
		for (int i = 0; i < C; i++)
			o << " " << distribution[i];
		o << endl;
		break;
	case SPLIT:
		o << " " << left << " " << right << " " << attr << " " << split_point
				<< endl;
		break;
	}
}

void tree_node::read(istream& i, int C) {
	int status_int;
	i >> status_int;
	status = NodeStatusType(status_int);
	assert(status != EMPTY);
	switch (status) {
	case TERMINAL:
		int label_int;
		i >> label_int;
		label = uchar(label_int);
		for (int j = 0; j < C; j++)
			i >> distribution[j];
		break;
	case SPLIT:
		i >> left >> right >> attr >> split_point;
		break;
	}
}

void tree_node_RC::write(ostream& o, int K, int C) const {
	// we shouldn't be saving any other kind of node
	assert(status == TERMINAL || status == SPLIT);
	o << int(status);
	switch (status) {
	case TERMINAL:
		o << " " << int(label);
		for (int i = 0; i < C; i++)
			o << " " << distribution[i];
		o << endl;
		break;
	case SPLIT:
		o << " " << left << " " << right;
		int i;
		for (i = 0; i < K; i++) {
			o << " " << atts[i];
		}
		for (i = 0; i < K; i++) {
			o << " " << weights[i];
		}
		o << " " << bias;
		o << endl;
		break;
	}
}

void tree_node_RC::read(istream& i, int K, int C) {
	int status_int;
	i >> status_int;
	status = NodeStatusType(status_int);
	assert(status != EMPTY);
	switch (status) {
	case TERMINAL:
		int label_int;
		i >> label_int;
		label = uchar(label_int);
		for (int j = 0; j < C; j++)
			i >> distribution[j];
		break;
	case SPLIT:
		i >> left >> right;
		int j;
		atts = new unsigned int[K];
		weights = new float[K];
		for (j = 0; j < K; j++) {
			i >> atts[j];
		}
		for (j = 0; j < K; j++) {
			i >> weights[j];
		}
		i >> bias;
		break;
	}
}

