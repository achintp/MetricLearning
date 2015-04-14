/*
 * File:   Thread.cpp
 * Author: Souvik Chatterjee
 *
 * Created on August 12, 2009, 2:54 AM
 */

#include "Thread.h"

namespace mom {
Thread::Thread(thread_start_t thread_start) {
	_thread_start = thread_start;
}
void Thread::Start(var_t thread_args) {
	pthread_create(&_thread, NULL, _thread_start, thread_args);
}
void Thread::Join() {
	pthread_join(_thread, NULL);
}
}
