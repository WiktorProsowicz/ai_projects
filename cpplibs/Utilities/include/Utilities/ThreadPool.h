#ifndef UTILITIES_INCLUDE_UTILITIES_THREADPOOL_HPP
#define UTILITIES_INCLUDE_UTILITIES_THREADPOOL_HPP

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <vector>

#include "Utilities/ThreadSafeQueue.hpp"

namespace utilities
{
class ThreadPool
{
public:
	ThreadPool() = default; /// Default constructor.

	/**
	 * @brief Creates a new thread pool and initializes it with `numThreads` of threads.
	 *
	 * @param numThreads Initial number of working threads.
	 */
	ThreadPool(size_t numThreads)
	{
		init(numThreads);
	}

	ThreadPool(const ThreadPool&) = delete;			   // Copy constructor
	ThreadPool(ThreadPool&&) = delete;				   // Move constructor
	ThreadPool& operator=(const ThreadPool&) = delete; // Copy assignment
	ThreadPool& operator=(ThreadPool&&) = delete;	   // Move assignment

	~ThreadPool()
	{
		terminate();
	}

public:
	/**
	 * @brief Initializes the thread pool with passed number of threads.
	 *
	 * @param numThreads Number of working threads created for the pool.
	 */
	void init(size_t numThreads);

	/**
	 * @brief Resizes the number of working threads. If the `numThreads` is smaller then size(), truncated
	 * threads shall bring their tasks to an end.
	 *
	 * @param numThreads
	 */
	void resize(size_t numThreads);

	/**
	 * @brief Joins the working threads and processes all of the available tasks.
	 *
	 */
	void terminate();

	/**
	 * @brief Joins the working threads and discards all waiting tasks.
	 *
	 */
	void cancel();

	/**
	 * @brief Tells if the thread pool has been initialized.
	 *
	 * @return true The pool is initialized and has prepared the threads.
	 * @return false The pool has not been yet initialized.
	 */
	bool initted() const
	{
		std::shared_lock<std::shared_mutex> lock(_mainMutex);
		return _initted;
	}

	/**
	 * @brief Tells if the pool can be provided with tasks and is able to process them.
	 *
	 * @return true Pool is active.
	 * @return false Pool is either stopped or not yet initialized.
	 */
	bool isRunning() const
	{
		return _isRunning();
	}

	/**
	 * @brief Returns the number of working threads.
	 *
	 * @return Number of thread.
	 */
	size_t size() const
	{
		std::shared_lock<std::shared_mutex> lock(_mainMutex);
		return _workers.size();
	}

	/**
	 * @brief Adds a new task to the queue.
	 *
	 * @tparam F Type of the function to be called within the task.
	 * @tparam Args Arguments to be packed with the function to create the task.
	 * @param f Callable.
	 * @param args Function arguments.
	 * @return Future object connected with the created task.
	 */
	template <class F, class... Args>
	auto addJob(F&& function, Args&&... args) const
	{
		using ReturnType = decltype(function(args...));
		using FutureType = std::future<ReturnType>;
		using PackagedTask = std::packaged_task<ReturnType()>;

		{
			std::shared_lock<std::shared_mutex> lock(_flagsMutex);
			if(_stopped || _cancelled)
			{
				throw std::runtime_error("Cannot add a new job to thread pool that has been terminated.");
			}
		}

		// NOLINTBEGIN
		auto boundFunction = std::bind(std::forward<F>(function), std::forward<Args>(args)...);
		// NOLINTEND

		auto task = std::make_shared<PackagedTask>(std::move(boundFunction));

		FutureType future = task->get_future();

		_tasks.emplace([task]() -> void { (*task)(); });

		_condition.notify_one();

		return future;
	}

private:
	/// Tells if the pool is active.
	bool _isRunning() const
	{
		std::shared_lock<std::shared_mutex> lock(_flagsMutex);
		return _initted && !_stopped && !_cancelled;
	}

	/// Creates new worker.
	void _spawn(size_t workerId);

private:
	mutable std::shared_mutex _mainMutex;
	std::vector<std::thread> _workers;

	std::shared_mutex _stopFlagsMutex;
	std::vector<bool> _stopFlags;

	mutable ThreadSafeQueue<std::function<void()>> _tasks;

	mutable std::shared_mutex _flagsMutex;
	bool _initted = false;
	bool _cancelled = false;
	bool _stopped = false;

	std::once_flag _once;
	mutable std::condition_variable_any _condition;
};
} // namespace utilities

#endif
