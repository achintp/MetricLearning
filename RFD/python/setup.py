import os
import sys

# determine python version on this machine
pyver = "python" + str(sys.version_info[0]) + "." + str(sys.version_info[1])

#build swig modules
print 'Building Swig Modules:'
#librf
print "building librf..."
os.system("swig -c++ -python src/librf.i")
os.system("g++ -march=native -fPIC -O3 -std=c++0x -c src/librf_wrap.cxx src/librf/*.cc src/librf/semaphores/*.cpp -I/usr/include/" + pyver)
# os.system("g++ -march=native -fPIC -ggdb3 -std=c++0x -c src/librf_wrap.cxx src/librf/*.cc src/librf/semaphores/*.cpp -I/usr/include/" + pyver)
os.system("g++ -march=native -fPIC -shared *.o -o src/_librf.so -lboost_thread")
os.system("rm *.o")
os.system("rm src/librf_wrap.*")
os.system("mv src/librf.py .")
os.system("mv src/_librf.so .")
