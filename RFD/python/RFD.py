import numpy as np
from librf import RandomForest

#first some utility code
def normalizelabels(labs):
    """
    Normalizes a set of input labels to a 1-dimensional ndarray of int32s, with labels ranging
    from 0 to n, where n is the number of different labels
    labs = 1-dimensional array-like of labels (will be flattened if not 1-dimensional)
    """
    labs = np.int32(np.array(labs)).flatten()
    labset = np.unique(labs)
    newlabs = np.arange(labset.shape[0])
    temp = np.zeros_like(labs)
    for i in xrange(labset.shape[0]):
        temp[labs == labset[i]] = newlabs[i]
    return temp

#now the actual random forest class (built on top of a SWIG binding to the C++ library)
class RF:
    """
    Class for training and applying a general random forest model.
    data = n by d numpy array containing the training data for the forest
    labels = n-length int vector containing a numerical label (from 0 to c-1, where
        c is the number of classes) for each point.
    forestloc = string describing the location of a random forest that has been saved
        to disk, note that if this is specified, then all other inputs (except threads)
        will be ignored
    ntrees = the number of trees to train (default 100)
    K = number of features to split on at each tree branch, or the number of different features to test, 
        depending on the algorithm (default is log2(d))
    minnodesize = minimum number of items in a leaf node - leaf nodes with fewer items than this will not
        be split, even if there are still items of different classes remaining in the node
    alg = determines the splitting algorithm used to train the forest - can be either 'single' or 'RC'.
        'single' tests k dimensions at each node, but chooses only one to split on, while 'RC' randomly
        chooses k dimensions to split on, tests F different random linear combinations of those
        dimensions and chooses the best one for the split
    F = determines the number of different thresholds to try for each variable (if alg='single'), or the 
        number of sets of random combination weights to try for each node (if alg='RC') (default = log2(n))
    uniquevals = if true, and algorithm is single, then the forest will precompute the unique values for each
        feature, and sample from those at each split, rather than directly from the elements present at the node.
        May give better/faster training for datasets where n >> the number of unique values for each feature, or
        where some non-outlier features values are very rare.
    threads = number of parallel threads to run - works by splitting up the requested number of trees
        among a number of forests equal to threads
    """
    def __init__(self, data=None, labels=None, forestloc=None, ntrees=100, K=None, minnodesize=1,
                 alg="single", F=None, uniquevals=False, threads=1):
        self.threads = threads
        # print labels
        # check to see if we are loading a forest from file or training a new one
        if forestloc == None:
            # initialize various class variables
            if type(data) != np.ndarray:
                raise ValueError("Invalid value for data, data must be a numpy ndarray.")
            if type(labels) != np.ndarray and labels != None:
                raise ValueError("Invalid value for labels, labels must be a numpy ndarray or None.")
            self.n = data.shape[0]
            self.d = data.shape[1]
            if labels != None:
                self.c = labels.max() + 1  # assume labels range from 0 to c-1
            self.ntrees = ntrees
            self.K = K
            self.F = F
            self.minnodesize = minnodesize
            if self.K == None:
                self.K = int(np.ceil(np.log2(self.d)))
            if self.F == None:
                self.F = int(np.ceil(np.log2(self.n)))
            # initialize our actual librf wrapper
            if alg == "single":
                splitfun = 0
            elif alg == "RC":
                splitfun = 1
            else:
                raise ValueError("Invalid splitting algorithm, valid options are 'single' and 'RC'.")
            if labels == None:
                raise ValueError("No training labels provided")
            elif len(labels.shape) == 1:
                self.rf = RandomForest(np.float32(data.T), normalizelabels(labels), self.ntrees,
                                       self.K, self.F, self.minnodesize, splitfun, self.threads, uniquevals)
        # if we are loading from a saved forest, ignore all that and just read from file
        else:
            self.rf = RandomForest()
            self.rf.read(forestloc)
            self.rf.setnthreads(self.threads)
            self.ntrees = self.rf.ntrees()
            self.d = self.rf.ndims()
            self.c = self.rf.C();
            
        
    def getvotes(self, testdata):
        """
        Applies the random forest to a set of input data and returns the response of each tree
        to each instance in the data.
        testdata = m by d ndarray containing the m input instances
        
        returns: an m by ntrees boolean array containing the responses of each tree to each
            test instance
        """
        if testdata.shape[1] != self.d:
            raise ValueError("Mistmatched array size - the given input array and the random forest have different dimensionalities.")
        out = np.zeros((testdata.shape[0], self.ntrees), dtype=np.byte)
        self.rf.votesset(np.float32(testdata), out)
        return np.uint8(out)
    
    def getregression(self, testdata):
        """
        Applies the random forest to a set of input data and returns the regression result for each point.
        testdata = m by d ndarray containing the m input instances
        
        returns: an m by c matrix containing the classification likelihood for each class
        """
        if testdata.shape[1] != self.d:
            raise ValueError("Mistmatched array size - the given input array and the random forest have different dimensionalities.")
        out = np.zeros((testdata.shape[0], self.c), dtype=np.float32)
        self.rf.regrset(np.float32(testdata), out)
        return out / self.ntrees
    
    def predict(self, testdata):
        """
        Applies the random forest to a set of input data and returns the resulting
        classification for each point.
        testdata = m by d ndarray containing the m input instances
        
        returns: a length-m array containing the classification of each test instance
        """
        temp = self.getregression(testdata)
        return np.int32(temp.argmax(axis=1))
    
    def variable_importance(self):
        """
        Computes the mean importance of each variable across all trees in the forest.  Variable
        importance within a tree is equal to the sum of the entropy reduction yielded by that variable.
        
        returns: a length-d array containing the mean importance of each variable
        """
        out = np.zeros(self.d, dtype=np.float32)
        self.rf.variable_importance(out)
        return out
    
    def save(self, forestloc):
        """
        Save the current forest to disk at the specified file location.
        """
        self.rf.write(forestloc)
        

#and now the RFD class, built on top of RF, but with some extra utilities        
class RFD:
    """
    Class represnting a random forest distance function.  Training is done on init.
    data = n by d array containing the data on which the forest will be learned, and which
        will be transformed into the new learned representation
    constraints = c by 3 numpy array containing the set of pairwise constraints, 
        where c is the number of constraints.  Constraints are listed in the 
        format [x1 x2 y], where x1 and x2 are the indices of the two data points 
        being constrained, and y is the relationship between them (either 0 for 
        a positive constraint (i.e. the two should be grouped together) or 1 
        for a negative constraint (i.e. the two should not be grouped together)).
    ntrees = number of trees to train (default is 100)
    minnodesize = the minimum number of instances to be stored in each leaf node
    alg = see description in RF
    F = see description in RF
    K = see description in RF
    forestloc = string denoting the location of a file the random forest
        should be loaded from - if this is given, other inputs (except threads)
        are ignored
    position = if True, then absolute (as well as relative) position is included in 
        the feature vector for each point pair (if false, only relative position is used)
    threads = number of parallel threads to use for training and testing
    kwargs = just a catch-all for any extraneous arguments that may find their way here
    """
    def __init__(self, data=None, constraints=None, ntrees=100, minnodesize=1, alg="single",
                 F=None, K=None, position=True, forestloc=None, threads=1, **kwargs):
        if forestloc != None:
            self.rf = RF(forestloc=forestloc, threads=threads)
        else:
            n, d = data.shape
            #generate constraint data points
            c1 = data[constraints[:, 0], :]
            c2 = data[constraints[:, 1], :]
            if position:
                features = np.zeros(dtype=np.float32, shape=(constraints.shape[0], d * 2))
                features[:, :d] = np.abs(c1 - c2)
                features[:, d:] = (c1 + c2) / 2
            else:
                features = np.float32(np.abs(c1 - c2))
            del c1, c2
            self.position = position
            self.rf = RF(data=features, labels=constraints[:, 2], ntrees=ntrees,
                                        minnodesize=minnodesize, alg=alg, F=F, K=K, threads=threads)
            
    def getdist(self, p1, p2):
        """
        Returns the random forest distance between the point sets p1 and p2
        p1 = m by d matrix containing a set of points
        p2 = m by d matrix containing a different set of points
        
        returns: an m-length vector containing the random forest distance from each point in
            p1 to the corresponding point in p2
        """
        if not np.all(p1.shape == p2.shape):
            raise ValueError("p1 and p2 must be the same shape.")
        d = p1.shape[1]
        if self.position:
            features = np.zeros(dtype=np.float32, shape=(p1.shape[0], d * 2))
            features[:, :d] = np.abs(p1 - p2)
            features[:, d:] = (p1 + p2) / 2
        else:
            features = np.float32(np.abs(p1 - p2))
        return self.rf.getregression(features)
    
    def saveforest(self, saveloc):
        """
        Saves the learned forest to file in the specified location.
        """
        self.rf.save(saveloc)
        
    def getdistmat(self, data1, data2):
        """
        Generates an inverted kernel matrix (i.e. distances rather than similarities) between 
        two datasets using the learned RFD as the distance function.
        data1 = n by d data matrix (instances will appear as rows in the distance matrix)
        data2 = m by d data matrix (instances will appear as columns in the distance matrix)
        
        returns: n by m float32 distance matrix between the two sets of input data
        """
        data1 = np.float32(data1)
        data2 = np.float32(data2)
        n = data1.shape[0]
        m = data2.shape[0]
        out = np.zeros(shape=(n, m), dtype=np.float32)
        self.rf.rf.rfdregr(data1, data2, out, self.position)
        # return out / self.rf.ntrees
        return out

    def getCodes(self, data):
         data = np.float32(data)
         n = data.shape[0]
         out = np.zeros(shape=(n, self.rf.rf.ntrees()), dtype = np.float32)
         self.rf.rf.rfdcode(data, out, self.position)
         return out
        

#convenience method for generating pairwise constraints from labels
def getconstraints(labels, numcs):
    """Generates a set of pairwise constraints from a label set.  Does not
    guarantee that no duplicate pairs occur, but does prevent points from being
    paired with themselves.  Assumes labels are from 0 to c-1, where c is the 
    number of classes.
    
    labels = n-length vector containing the class labels for each point
    numcs = the number of positive constraints to generate per class.  Note that an equal
        number of negative constraints are also created.
        
    returns a numcs*2*numclasses by 3 array, where the first two columns are point indices
        and the third is either 0 (for must-link constraints) or 1 (for cannot-link
        constraints)
    """
    labset = np.unique(labels)
    c = labset.shape[0]
    cs = np.zeros((numcs * 2 * c, 3), dtype=np.uint32)
    for i in xrange(c):
        #get positive constraints
        thisclass = np.nonzero(labels == labset[i])[0]
        otherclass = np.nonzero(labels != labset[i])[0]
        c1 = np.random.randint(thisclass.shape[0], size=numcs)
        c2 = np.random.randint(thisclass.shape[0], size=numcs)
        #ensure no point is constrained with itself
        done = False
        while(not(done)):
            tmp = c1 == c2
            numsame = tmp.sum()
            if numsame == 0:
                done = True
            else:
                c2[tmp] = np.random.randint(thisclass.shape[0], size=numsame)
        c3 = np.zeros(numcs)
        cs[i * numcs * 2:i * numcs * 2 + numcs] = np.vstack((thisclass[c1], thisclass[c2], c3)).T
        #get negative constraints
        c1 = np.random.randint(thisclass.shape[0], size=numcs)
        c2 = np.random.randint(otherclass.shape[0], size=numcs)
        c3 = np.ones(numcs)
        cs[i * numcs * 2 + numcs:i * numcs * 2 + numcs * 2] = np.vstack((thisclass[c1], otherclass[c2], c3)).T
    return cs
