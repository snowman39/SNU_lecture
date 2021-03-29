//------------------------------------------------------------------------------
//
// memtrace
//
// trace calls to the dynamic memory manager
//
#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <memlog.h>
#include <memlist.h>

//
// function pointers to stdlib's memory management functions
//
static void *(*mallocp)(size_t size) = NULL;
static void (*freep)(void *ptr) = NULL;
static void *(*callocp)(size_t nmemb, size_t size);
static void *(*reallocp)(void *ptr, size_t size);

//
// statistics & other global variables
//
static unsigned long n_malloc  = 0;
static unsigned long n_calloc  = 0;
static unsigned long n_realloc = 0;
static unsigned long n_allocb  = 0;
static unsigned long n_freeb   = 0;
static item *list = NULL;

void *malloc(size_t size)
{
	char *error;
	void *ptr;

	if (!mallocp) {
		mallocp = dlsym(RTLD_NEXT, "malloc");
		if ((error = dlerror()) != NULL) {
			fputs(error, stderr);
			exit(1);
		}
	}

	ptr = mallocp(size);
	LOG_MALLOC(size, ptr);
	alloc(list, ptr, size);
	n_malloc++;
	n_allocb += size;

	return ptr;
}

void *calloc(size_t nmemb, size_t size)
{
	char *error;
	void *ptr;

	if (!callocp) {
		callocp = dlsym(RTLD_NEXT, "calloc");
		if ((error = dlerror()) != NULL) {
			fputs(error, stderr);
			exit(1);
		}
	}

	ptr = callocp(nmemb, size);
	LOG_CALLOC(nmemb, size, ptr);
	alloc(list, ptr, nmemb*size);
	n_calloc++;
	n_allocb += nmemb * size;

	return ptr;
}

void *realloc(void *ptr, size_t size)
{
	char *error;
	if (!reallocp) {
		reallocp = dlsym(RTLD_NEXT, "realloc");
		if ((error = dlerror()) != NULL) {
			fputs(error, stderr);
			exit(1);
		}
	}
	
  item *old_block = find(list, ptr);
	void *res = ptr;

	if (old_block != NULL) {
		if (old_block->cnt) {
			res = reallocp(ptr, size);
			LOG_REALLOC(ptr, size, res);
			if (old_block->size < size) {
				dealloc(list, ptr);
				n_freeb += old_block->size;
				alloc(list, res, size);
				n_realloc++;
				n_allocb += size;
			}
		} else {
			// DOUBLE FREE
			res = reallocp(NULL, size);
			alloc(list, res, size);
			LOG_REALLOC(ptr, size, res);
			LOG_DOUBLE_FREE();
			n_realloc++;
			n_allocb += size;
		}
	} else {
		// ILLEGAL FREE
		res = reallocp(NULL, size);
		alloc(list, res, size);
		LOG_REALLOC(ptr, size, res);
		LOG_ILL_FREE();
		n_realloc++;
		n_allocb += size;
	}

	return res;
}

void free(void *ptr)
{
	char *error;
	
	if (!freep) {
		freep = dlsym(RTLD_NEXT, "free");
		if ((error = dlerror()) != NULL) {
			fputs(error, stderr);
			exit(1);
		}
	}
	
	if (ptr != NULL) {
		LOG_FREE(ptr);
		item *target_block = find(list, ptr);
		if (target_block != NULL) {
			if (target_block->cnt) {
				freep(ptr);
				item *freed_item = dealloc(list, ptr);
				n_freeb += freed_item->size;
			} else {
				LOG_DOUBLE_FREE();	
			}
		} else {
			LOG_ILL_FREE();
		}
	}
}

//
// init - this function is called once when the shared library is loaded
//
__attribute__((constructor))
void init(void)
{
  char *error;

  LOG_START();

  // initialize a new list to keep track of all memory (de-)allocations
  // (not needed for part 1)
  list = new_list();

  // ...
}

//
// fini - this function is called once when the shared library is unloaded
//
__attribute__((destructor))
void fini(void)
{
  // ...
	unsigned long n_alloc = n_malloc + n_calloc + n_realloc;
	unsigned long avg_allocb = 0;
	if (n_alloc)
		avg_allocb = n_allocb / n_alloc;

  LOG_STATISTICS(n_allocb, avg_allocb, n_freeb);
	
	if (n_allocb > n_freeb) {
		LOG_NONFREED_START();
		item* block = list->next;
		while (block != NULL) {
			if (block->cnt > 0) 
				LOG_BLOCK(block->ptr, block->size, block->cnt);
			block = block->next;
		}
	}
  LOG_STOP();

  // free list (not needed for part 1)
  free_list(list);
}

// ...
