/*
 * Includes a set of common object definitions and utility functions potentially used by multiple libraries
 */

#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>
#include <algorithm>
#include <boost/functional/hash.hpp>

using namespace std;

//define functions for hashing/comparing pairs
typedef boost::hash<pair<int, int>> pair_hash;
template<class T> struct pair_eq
{
    bool operator()(const pair<const T, const T>& a, const pair<const T, const T>& b) const
    {
        return (a.first == b.first && a.second == b.second);
    }
};

//Structure for representing the closure of a set of pairwise constraints.
//The structure members collectively allow the known semantic relationships
//between points to be tracked efficiently.
class closure
{
public:
    unordered_map<int, int> setmembership; //<point id, set id>
    unordered_map<int, unordered_set<int> > pointsets; //<set id, set of <point id>>
    vector<pair<int, int> > neglinks; //<set id 1, set id 2> (where set1 and set2 are negatively linked)
};

//maintains a set of negative links for faster lookup of relations between specific point pairs
//this is not needed in subclosures, which rely strictly on the vector of negative links (which can be
//constructed and iterated more efficiently than the set)
class topclosure: public closure
{
public:
    unordered_set<pair<int, int>, pair_hash, pair_eq<int> > neglinkset;
};

typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uchar;
typedef unsigned char byte;

//Select K random, unique values from the range 0 to n-1 and store them in the vector v
//note that v is not cleared before appending the new values
static void random_sample(int n, int K, vector<int>* v, unsigned int* seed)
{
    if (K < n)
    {
        //first create shuffled array
        int* temp = new int[n];
        int i, j;
        temp[0] = 0;
        for (i = 1; i < n; i++)
        {
            j = rand_r(seed) % (i + 1);
            temp[i] = temp[j];
            temp[j] = i;
        }
        //now take first K points from it
        for (i = 0; i < K; i++)
        {
            v->push_back(temp[i]);
        }
        delete temp;
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            v->push_back(i);
        }
    }
}

//Randomly permute the values 0 to n-1 and store the permuted list in vector v
static void random_permute(int n, vector<int>& v, unsigned int* seed)
{
    v.clear();
    v.resize(n);
    int i, j;
    v[0] = 0;
    for (i = 1; i < n; i++)
    {
        j = rand_r(seed) % (i + 1);
        v[i] = v[j];
        v[j] = i;
    }
}

//Randomly permute the values already found in vector v
static void random_permute(vector<int>& v, unsigned int* seed)
{
    int i, j, n, temp;
    n = v.size();
    for (i = n - 1; i > 0; i--)
    {
        j = rand_r(seed) % (i + 1);
        temp = v[j];
        v[j] = v[i];
        v[i] = temp;
    }
}

static float rand_float(float a, float b, unsigned int* seed)
{
    return ((b - a) * ((float) rand_r(seed) / RAND_MAX)) + a;
}

//argsort an input vector and return a pointer (under the control of the calling
//function) to a vector of the sorted indices
template<typename T>
vector<int>* sort_indexes(const vector<T>& v)
{

    // initialize original index locations
    vector<int>* idx = new vector<int>(v.size());
    for (int i = 0; i != idx->size(); i++)
        idx[i] = i;

    // sort indexes based on comparing values in v
    sort(idx->begin(), idx->end(), [&v](int i1, int i2)
    {   return v[i1] < v[i2];});

    return idx;
}

#endif
