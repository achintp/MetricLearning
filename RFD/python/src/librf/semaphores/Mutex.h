/*
 * File:   Mutex.h
 * Author: Souvik Chatterjee
 * This file declares the interface for the Mutex class
 * Mutex class is a wrapper over the pthread mutex.
 * It provides an C# like object oriented implementation
 * of unix pthread mutex
 */

#ifndef _MUTEX_H
#define    _MUTEX_H
#include <pthread.h>

class Mutex {

public:
	//Mutext::Lock() gains a lock on the MUTEX
	void Lock();

	//Mutext::Unlock() releases the MUTEX
	void Unlock();

private:
	//unix pthread instance
	pthread_mutex_t _mutex;
};

#endif    /* _MUTEX_H */
