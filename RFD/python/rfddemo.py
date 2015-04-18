import numpy as np
from scipy.io import savemat, loadmat
from RFD import *
import pprint
from librf import RandomForest


def getCodes(data, rf):
         data = np.float32(data)
         n = data.shape[0]
         out = np.zeros(shape=(n, rf.ntrees()), dtype = np.float64)
         rf.rfdcode(data, out, False)
         return out


np.set_printoptions(precision=4, suppress=True)

temp = loadmat("iris.mat")
dat = temp["da_iris"]
# pprint.pprint(dat)
# print len(dat)
# print temp["la_iris"]
lab = normalizelabels(temp["la_iris"])

# cons = getconstraints(lab, 100)
savemat('labs.mat',{'labs':lab, 'dat':dat})
# print cons
# rfd = RFD(dat,#data
#           cons, #constraints
#           500, #number of trees
#           5, #minimum node size
#           "single", #splitting function type
#           None, #number of candidate thresholds to test per split feature (using default)
#           None, #number of different candidate features to test per split (using default)
#           False, #use absolute position information
#           None, #saved forest location (None if training new forest)
#           8) #number of threads

rf = RandomForest(np.float32(dat).T,
		lab,
		2,
		2,
		8,
		1,
		0,
		1,
		False
		)


codes = getCodes(dat, rf)
# dists = rfd.getdistmat(dat, dat)
# codes = rfd.getCodes(dat)
savemat('test.mat', {'codes': codes})
# for item in codes:
# 	pprint.pprint(item)

# for item in dists:
# 	print len(item)
# 	pprint.pprint(item)
# print len(dists)