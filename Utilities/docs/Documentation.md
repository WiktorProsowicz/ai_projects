# Introduction

Utilities lib contain project-wide classes and functions helping at development.

# Version changes


## 1.0.0

- Introduced [ThreadPool](#ThreadPool) and [ThreadSafeQueue](#ThreadSafeQueue)

# Components

## ThreadPool

Class responsible for managing threads and assigning tasks to them, ensuring safety while calling its methods. ThreadPool uses [ThreadSafeQueue](#ThreadSafeQueue) for the main safety-check for the storing/obtaining assigned tasks. Each of the worker threads is put to sleep in case there are no available tasks at the moment. Additionally the pool is capable of cancelling individual threads at request.

Implementation
```cpp
namespace utilities
{
    class ThreadPool;
}
```

Main workflow of the ThreadPool consists of following steps:
- ThreadPool is created and initialized with a desired number of threads.
- Tasks are added to the pool and wait for the threads to pop them from the queue.
- The pool can be resized and in case the number of threads is decreased - truncated threads perform at most one next task.
- The pool is either terminated (all left tasks are run) or cancelled (left tasks are dismissed). Threads are joined.

Example:
```cpp

static constexpr size_t kNumThreads = 4;

utilities::ThreadPool pool;

pool.init(kNumThreads);

assert(pool.size() == 4, "Wrong size");
assert(pool.initted(), "Pool should be initted be now.");
assert(pool.isRunning(), "Pool should be running be now.");

for(const size_t number : {0, 1, 2, 3, 4, 5, 6, 7})
{
    pool.addJob([](size_t n){ std::cout << n << std::endl; }, number);
}

pool.resize(2);
assert(pool.size() == 2, "Wrong size");

pool.terminate();

assert(!pool.isRunning(), "Pool should not be running.");

// By now all of the number should have been printed
// If the `cancel()` was called instead of `terminate()` - not necessarily
```

## ThreadSafeQueue

Template class wrapping std::queue. Ensures thread-safety while accessing stored elements.

Implementation:
```cpp
namespace utilities
{
    template <typename T>
    class ThreadSafeQueue : protected std::queue<T>;
}
```
