/*
 * File:   Mutex.cpp
 * Author: Souvik Chatterjee
 * This CPP File Contains the implementation of the header Mutex.h
 */

#include <stdio.h>
#include <ios>
#include <pthread.h>

#include "Mutex.h"
//---------------------------------------
/*
 * Mutext::Lock() gains a lock on the MUTEX
 */
void Mutex::Lock() {
	//execute pthread mutex lock system call
	//with member pthread mutext instance
	//pass the reference of the pthread mutex instance
	pthread_mutex_lock(&_mutex);
}

//--------------------------------------
/*
 * Mutext::Unlock() releases the MUTEX
 */
void Mutex::Unlock() {
	//execute pthread mutex unlock system call
	//with member pthread mutext instance
	//pass the reference of the pthread mutex instance
	pthread_mutex_unlock(&_mutex);
}
//--------------------------------------
