/*
 * File:   Semaphore.h
 * Author: Souvik Chatterjee
 * This class is a C# like object oriented wrapper
 * for local(in process) and shared(system wide) semaphore
 * This file defines the interface for the Semaphore class
 *
 * Important: I designed the class inspired
 * by the .NET framework's implementation of Semaphore
 * In .Net framework, apart from Mutex there is a Semaphore class
 * which can be used as both local and shared
 * semaphores. I am not sure of .Net internal implementation.
 * But this C++ class internally uses two completely
 * different implementation UNIX semget C system call
 * and UNIX pthread C system call for shared and local semaphores
 * respectively.
 */

#ifndef _SEMAPHORE_H
#define    _SEMAPHORE_H

namespace mom {
class Semaphore {
public:
	//This constructor creates(or retrieves existing)
	//system wide semaphore with the given sem_id
	//using this call 65535 different system wide semaphores can be created
	//this shared semaphore is implemented with UNIX semget C system call
	Semaphore(unsigned short sem_id, unsigned int initial_count,
			unsigned int max_count);

	//This constructor creates local(in process) semaphore
	//in UNIX local semaphore in a MUTEX.
	//this local semaphore internally uses UNIX pthread mutext
	Semaphore(int initial_count, int max_count);

	//waits until succeeds to acquire a room in the semaphore object
	void WaitOne();
	//releases one room among the acquired rooms in the semaphore objet
	void Release();
	//releases specified number(release_count) rooms
	//among the acquired rooms in the semaphore objet
	void Release(int release_count);
	//destructor
	virtual ~Semaphore();

private:
	//internal flag to determine if the created semaphore is local or shared
	bool _is_local;
	//internal handle for the created semaphore instance
	void* _semaphore_instance_ptr;
};
}

#endif    /* _SEMAPHORE_H */
