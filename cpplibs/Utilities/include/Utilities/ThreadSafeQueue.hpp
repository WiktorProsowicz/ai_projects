#ifndef UTILITIES_INCLUDE_UTILITIES_THREADSAFEQUEUE_HPP
#define UTILITIES_INCLUDE_UTILITIES_THREADSAFEQUEUE_HPP

// __C++ standard headers__
#include <shared_mutex>
#include <mutex>
#include <queue>

namespace utilities
{
/**
 * @brief Class serving as a FIFO data structure, delegating the getting/setting requests to the underlying
 * standard queue and protecting the contained objects by multiple-thread access.
 *
 * @tparam T Type of the stored objects.
 */
template <typename T>
class ThreadSafeQueue : protected std::queue<T>
{
public:
	/**
	 * @brief Creates a new empty queue.
	 *
	 */
	ThreadSafeQueue() = default;

	ThreadSafeQueue(const ThreadSafeQueue&) = delete;			 // Copy constructor
	ThreadSafeQueue(ThreadSafeQueue&&) = delete;				 // Move constructor
	ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete; // Copy assignment
	ThreadSafeQueue& operator=(ThreadSafeQueue&&) = delete;		 // Move assignment

	/**
	 * @brief Destroys the queue, deleting all of the contained objects.
	 *
	 */
	~ThreadSafeQueue()
	{
		clear();
	}

public:
	/// Tells if the queue is empty.
	bool empty() const
	{
		std::shared_lock<std::shared_mutex> lock(mutex_);
		return std::queue<T>::empty();
	}

	/// Tells the number of contained elements.
	size_t size() const
	{
		std::shared_lock<std::shared_mutex> lock(mutex_);
		return std::queue<T>::size();
	}

	/// Erases the contained elements.
	void clear()
	{
		std::unique_lock<std::shared_mutex> lock(mutex_);

		while(!std::queue<T>::empty())
		{
			std::queue<T>::pop();
		}
	}

	/// Adds the `object` to the queue.
	void push(const T& object)
	{
		std::unique_lock<std::shared_mutex> lock(mutex_);
		std::queue<T>::push(object);
	}

	/// Creates a new element from given `args`.
	template <typename... Args>
	void emplace(Args&&... args)
	{
		std::unique_lock<std::shared_mutex> lock(mutex_);
		std::queue<T>::emplace(std::forward<Args>(args)...);
	}

	/**
	 * @brief Attempts to erase the front element of the queue and assigns it to the provided holder.
	 *
	 * @param holder Object to which the front element will be assigned.
	 * @return true
	 * @return false
	 */
	bool tryPop(T& holder)
	{
		std::unique_lock<std::shared_mutex> lock(mutex_);

		if(std::queue<T>::empty())
		{
			return false;
		}

		holder = std::move(std::queue<T>::front());
		std::queue<T>::pop();
		return true;
	}

private:
	mutable std::shared_mutex mutex_{};
};
} // namespace utilities

#endif
