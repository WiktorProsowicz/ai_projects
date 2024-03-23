# Introduction

Utilities lib contain project-wide classes and functions helping at development.

# Version changes


## 1.0.0

- Introduced [ThreadPool](#threadpool) and [ThreadSafeQueue](#threadsafequeue)

## 1.1.0

- Introduced [SerializationPack](#serializationpack) and basic object-to-byte algorithms (Numerics, Vectors, C-strings, Cpp-strings)

# Components

## ThreadPool

Class responsible for managing threads and assigning tasks to them, ensuring safety while calling its methods. ThreadPool uses [ThreadSafeQueue](#threadsafequeue) for the main safety-check for the storing/obtaining assigned tasks. Each of the worker threads is put to sleep in case there are no available tasks at the moment. Additionally the pool is capable of cancelling individual threads at request.

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

## SerializationPack

Class used to enclose arguments of various types and store them in type-erased form. When streamed via `<<` operator, the objects are unpacked and casted to initial types and serialized into bytes form using custom mechanics of converting elements into binary form. For example `std::string` instance is represented as its internal char-string rather than the direct memory of the instance.

Implementation:

```cpp
namespace utilities
{
    template <typename... ArgTypes>
    class SerializationPack;
}
```

SerializationPack can be streamed:

```cpp
std::stringstream ss;

ss << utilities::SerializationPack(
    "abcd",
    uint32_t(0),
    std::vector<std::string>{"ab", "def"}
);

// ss contains: 0x61 0x62 0x63 0x64 0x00 0x00 0x00 0x00 0x61 0x62 0x64 0x65 0x66
```


Currently supported types for conversion:
- Numeric built-in types (`uint32_t`, `int64_t`, `double`, etc).
- C-style strings.
- `std::string` instances.
- `std::vector` instances holding any other convertible type.
