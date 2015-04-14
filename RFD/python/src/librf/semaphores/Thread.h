/*
 * File:   Thread.h
 * Author: Souvik Chatterjee
 *
 * Created on August 12, 2009, 2:54 AM
 */

#ifndef _THREAD_H
#define    _THREAD_H

#include<pthread.h>
#include <stdio.h>
#include <ios>

namespace mom {
typedef void* var_t;
typedef var_t (*thread_start_t)(var_t);

class Thread {
public:
	Thread(thread_start_t thread_start);
	void Start(var_t thread_args);
	void Join();
	static int Sleep(unsigned long millisecs) {
		long sec = (long) (millisecs / 1000);
		long nsec = (millisecs - (sec * 1000)) * 1000;
		timespec delay = { sec, nsec };
		int return_val = nanosleep(&delay, (timespec*) NULL);
		return return_val;
	}

private:
	thread_start_t _thread_start;
	pthread_t _thread;
};
}
#endif    /* _THREAD_H */
