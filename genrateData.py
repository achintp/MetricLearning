import sys
import pprint
import numpy as np


def generateGaussian(mean, var, size):
    t = np.random.normal(mean, var, size)
    assert(abs(mean - np.mean(t)) < 0.01)
    pprint.pprint(t)
    print np.mean(t)

if __name__ == '__main__':
    generateGaussian(float(sys.argv[1]),
                     float(sys.argv[2]),
                     int(sys.argv[3]))
