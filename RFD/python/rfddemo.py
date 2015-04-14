import numpy as np
from scipy.io import savemat, loadmat
from RFD import *
import pprint

np.set_printoptions(precision=4, suppress=True)

temp = loadmat("iris.mat")
dat = temp["da_iris"]
print len(dat)
print temp["la_iris"]
lab = normalizelabels(temp["la_iris"])

cons = getconstraints(lab, 100)
print cons
rfd = RFD(dat,#data
          cons, #constraints
          500, #number of trees
          5, #minimum node size
          "single", #splitting function type
          None, #number of candidate thresholds to test per split feature (using default)
          None, #number of different candidate features to test per split (using default)
          False, #use absolute position information
          None, #saved forest location (None if training new forest)
          8) #number of threads

# dists = rfd.getdistmat(dat, dat)
codes = rfd.getCodes(dat)
for item in codes:
	pprint.pprint(item)

# for item in dists:
# 	print len(item)
# 	pprint.pprint(item)
# print len(dists)