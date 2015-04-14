/*
 * File:   SemaphoreWrapper.cpp
 * Author: Souvik Chatterjee
 * This file defines the implementation
 * of the Semaphore interface declared in Semaphore.h
 * Semaphore class uses __shared_semaphore class
 * for shared(system wide) semaphore which is basically an object
 * oriented wrapper over UNIX semget.
 * On the other hand, it uses __local_semaphore class
 * for local(in process) semaphore which is basically an object
 * oriented wrapper over UNIX pthread mutex.
 * Reference: to know understand the UNIX system calls
 * used throughout this implementation please refer to
 * Open Group (http://www.unix.org/single_unix_specification/) and
 * The Linux Programmer's Guide (http://tldp.org/LDP/lpg/)
 */

#include "Semaphore.h"
//#include "Mutex.h"
#include "Thread.h"
#include <iostream>
#include <stdlib.h>
#include <sys/sem.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <fcntl.h>
#include <errno.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

namespace mom {
//------------------- System wide shared semaphore
//                   (class __shared_semaphore)-------------------
class __shared_semaphore {
public:
	// creates (or retrieves existing) system wide(shared)
	// semaphore with given semaphore id(sem_id)
	// initial_count refers to the initially available
	// rooms that can be acquired by _wait_one method
	// max_count refers to the maximum number of rooms
	// that can be acquired by _wait_one_method
	__shared_semaphore(unsigned short sem_id, unsigned int initial_count,
			unsigned int max_count);

	// waits untils succeeds to acquire a room in the semaphore object
	void _wait_one();
	// releases specified number(release_count) of rooms
	// among the acquired rooms in the semaphore object
	// if no rooms are currently occupied, it simply ignores
	// the call. you an implement it with a custom exception
	// thrown
	void _release(unsigned int release_count);

private:
	// releases specified number(release_count) of rooms
	// among the acquired rooms in the semaphore object
	// it can not provided rooms exceeding the <max_count>.
	// any such attempt will be simply ignored. you can
	// implement this with a custom xception thrown
	void _release_internal(unsigned int release_count);
	//holds the key of the shared semaphore object
	key_t _sem_key;
	//holds the maximum count for the semaphore object
	unsigned int _max_count;
	//holds the semaphore id retrieved from the system
	int _sem_id;
};

__shared_semaphore::__shared_semaphore(unsigned short sem_id,
		unsigned int initial_count, unsigned int max_count) {
	//set the key
	_sem_key = (key_t) sem_id;
	//set max count
	_max_count = max_count;
	//set wait instruction

	//set sem id to not set i.e. -1
	_sem_id = -1;

	//define the semaphore creation mode
	// IPC_CREATE: create a new semaphore if already
	// there is no sem_id associated with sem_key
	// IPC_EXCL: the semget function will fail if there
	// is already sem_id exists associated with the sem_key
	// S_IRUSR: owner has the read permission on semaphore object
	// S_IWUSR: owner has the write permission on semaphore object
	// S_IROTH: read permission on semaphore object for others
	// S_IWOTH: write permission on semaphore object for others
	mode_t sem_mode = IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR | S_IRGRP
			| S_IWGRP | S_IROTH | S_IWOTH;

	//lets try to retrieve the semaphore id for
	//the existing semaphore(if any) associated with the sem_key
	//it will return -1 if there is no semaphore
	//available in the system associated with the given key
	_sem_id = semget(_sem_key, 0, 0);

	if (_sem_id == -1) { //okay no semaphore found in the system for the given key
		//now lets create a new semaphore with the sem_key and with sem_mode
		_sem_id = semget(_sem_key, 1, sem_mode);
		//lets assume it failed due to some reason..
		//if you use this code, I will recommend to use
		//proper object oriented exception handling here
		if (_sem_id == -1) {
			if (errno == EEXIST) {
				perror("IPC error 1: semget");
			} else {
				perror("IPC error 2: semget");
			}
			exit(1);
		}
		//this process created the semaphore first
		//lets provide <initial_count> number of rooms
		_release_internal(initial_count);
	}
}

void __shared_semaphore::_wait_one() {

	sembuf sem_instruction;
	sem_instruction.sem_num = 0;
	sem_instruction.sem_op = -1;
	sem_instruction.sem_flg = SEM_UNDO;
	//execute the semop system call on the semaphore
	//with the prepared wait instruction
	if (semop(_sem_id, &sem_instruction, 1) != -1) {
		//for proper functionality, this line of code is required
		//it sets the semaphore's current value
		//in the system which other process can feel
		//i am not very much sure why it is required.
		//I used it after doing a lots of debugging
		//please put a comment in the article
		//if you have the detailed information for it
		semctl(_sem_id, 0, SETVAL, semctl(_sem_id, 0, GETVAL, 0));
	}
}

void __shared_semaphore::_release(unsigned int release_count) {

	if (semctl(_sem_id, 0, GETNCNT, 0) > 0) {
		//if atleast one process is waiting for a resource
		_release_internal(release_count);
	} else {
		//no process is waiting fo the resource..
		//so simply ignored the call.. you should throw some
		// custom exception from here
	}
}

void __shared_semaphore::_release_internal(unsigned int release_count) {

	if (semctl(_sem_id, 0, GETVAL, 0) < _max_count) {
		sembuf sem_instruction;
		sem_instruction.sem_num = 0;
		sem_instruction.sem_op = release_count;
		sem_instruction.sem_flg = IPC_NOWAIT | SEM_UNDO;
		//execute the semop system call on the semaphore
		//with the prepared signal instruction
		if (semop(_sem_id, &sem_instruction, 1) != -1) {
			//for proper functionality, this line of code is required
			//it sets the semaphore's current value
			//in the system which other process can feel
			//i am not very much sure why it is required.
			//I used it after doing a lots of debugging
			//please put a comment in the article
			//if you have the detailed information for it
			semctl(_sem_id, 0, SETVAL, semctl(_sem_id, 0, GETVAL, 0));
		}
	} else {
		//ignored the call. you should thorw some custo exception
	}
}

//----------------------- Local semaphore --------------(class __local_semaphore)--------
class __local_semaphore {
public:
	//creates a logical couting semaphore using mutex.
	//This semaphore has no scope out side the process
	//inwhich its running. so it can be used
	//for inter-thread signalling but not interprocess signalling
	__local_semaphore(int initial_count, int max_count);
	//~__local_semaphore();
	void _wait_one();
	void _release(int release_count);

private:
	unsigned int _initial_count;
	unsigned int _max_count;
	boost::mutex _wait_handle;
	//boost::mutex* _local_handle;
	boost::condition_variable _cond;
};

__local_semaphore::__local_semaphore(int initial_count, int max_count) {
	_initial_count = initial_count;
	_max_count = max_count;
	//_wait_handle = new boost::mutex();
	//_local_handle = new boost::mutex();
}

/*__local_semaphore::~__local_semaphore() {
 std::cout << "test3" << std::endl << std::flush;
 delete _local_handle;
 std::cout << "test4" << std::endl << std::flush;
 delete _wait_handle;
 std::cout << "test5" << std::endl << std::flush;
 }*/

void __local_semaphore::_wait_one() {
	boost::mutex::scoped_lock l(_wait_handle);
	_initial_count++;
	while (_initial_count > _max_count)
		_cond.wait(l);
}

void __local_semaphore::_release(int release_count) {
	boost::mutex::scoped_lock l(_wait_handle);
	_initial_count -= release_count;
	if (_initial_count <= _max_count)
		_cond.notify_one();
}

//----------------------- Semphore (wrapper)------------(class Semaphore)----
//create a system wide semaphore with the sem_id provided
Semaphore::Semaphore(unsigned short sem_id, unsigned int initial_count,
		unsigned int max_count) {
	_is_local = false;
	__shared_semaphore* shared_semaphore =
	new __shared_semaphore(sem_id, initial_count, max_count);
	_semaphore_instance_ptr = (void*)shared_semaphore;
}

//create a local semaphore
			Semaphore::Semaphore(int initial_count, int max_count) {
				_is_local = true;
				__local_semaphore* local_semaphore =
				new __local_semaphore(initial_count, max_count);
				_semaphore_instance_ptr = (void*)local_semaphore;
			}

			//block the caller until it succeeds to occupy a room
			void Semaphore::WaitOne() {
				if(_is_local) {
					((__local_semaphore*)_semaphore_instance_ptr)->_wait_one();
				}
				else {
					((__shared_semaphore*)_semaphore_instance_ptr)->_wait_one();
				}
			}

			//release <release_count> occupied rooms
			void Semaphore::Release(int release_count) {
				if(_is_local) {
					((__local_semaphore*)_semaphore_instance_ptr)->_release(release_count);
				}
				else {
					((__shared_semaphore*)_semaphore_instance_ptr)->_release(release_count);
				}
			}

			//release an occupied room
			void Semaphore::Release() {
				Release(1);
			}

			Semaphore ::~Semaphore() {
				if(_is_local) {
					__local_semaphore* __semaphore_ptr =
					(__local_semaphore*)_semaphore_instance_ptr;
					delete __semaphore_ptr;
				}
				else {
					__shared_semaphore* __semaphore_ptr =
					(__shared_semaphore*)_semaphore_instance_ptr;
					delete __semaphore_ptr;
				}
				_semaphore_instance_ptr = NULL;
			}
			//-----------------------------------------------------------------------
		}
