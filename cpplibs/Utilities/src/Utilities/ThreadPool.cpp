#include "Utilities/ThreadPool.h"

#include <functional>

namespace utilities
{
void ThreadPool::init(size_t numThreads)
{
	std::call_once(once_,
				   [this, &numThreads]
				   {
					   initted_ = true;

					   resize(numThreads);
				   });
}

void ThreadPool::terminate()
{
	{
		if(_isRunning())
		{
			std::unique_lock<std::shared_mutex> lock(flagsMutex_);
			stopped_ = true;
		}
		else
		{
			return;
		}
	}

	condition_.notify_all();

	for(auto& worker : workers_)
	{
		worker.join();
	}
}

void ThreadPool::cancel()
{
	{
		std::unique_lock<std::shared_mutex> lock(mainMutex_);
		if(_isRunning())
		{
			cancelled_ = true;
		}
		else
		{
			return;
		}
	}

	tasks_.clear();
	condition_.notify_all();

	for(auto& worker : workers_)
	{
		worker.join();
	}

	workers_.clear();
}

void ThreadPool::_spawn(size_t workerId)
{
	while(true)
	{
		bool obtainedATask = false;
		std::function<void()> runTask;

		{
			std::unique_lock<std::shared_mutex> lock(mainMutex_);

			condition_.wait(lock,
							[this, &obtainedATask, &runTask, &workerId]
							{
								obtainedATask = tasks_.tryPop(runTask);

								std::shared_lock<std::shared_mutex> flagsLock(flagsMutex_);
								std::shared_lock<std::shared_mutex> stopFlagsLock(stopFlagsMutex_);

								return cancelled_ || stopped_ || obtainedATask || stopFlags_.at(workerId);
							});
		}

		{
			std::shared_lock<std::shared_mutex> flagsLock(flagsMutex_);

			// In case the whole pool has been cancelled - don't care about the following task
			// In case the pool has been stopped - run tasks until there are any
			if(cancelled_ || (stopped_ && !obtainedATask))
			{
				return;
			}
		}

		if(obtainedATask)
		{
			runTask();
		}

		{
			std::shared_lock<std::shared_mutex> stopFlagsLock(stopFlagsMutex_);

			// Closing the individual thread only after possible processing of the task
			if(stopFlags_.at(workerId))
			{
				return;
			}
		}
	}
}

void ThreadPool::resize(size_t numThreads)
{
	if(!_isRunning())
	{
		throw std::runtime_error("Cannot resize thread pool which is not running.");
	}

	if(numThreads < size())
	{
		{
			std::unique_lock<std::shared_mutex> stopFlagsMutex(stopFlagsMutex_);

			for(size_t threadNum = numThreads; threadNum < workers_.size(); threadNum++)
			{
				stopFlags_.at(threadNum) = true;
			}
		}

		condition_.notify_all();

		for(size_t threadNum = numThreads; threadNum < workers_.size(); threadNum++)
		{
			workers_.at(threadNum).join();
		}

		workers_.resize(numThreads);
		stopFlags_.resize(numThreads);

		return;
	}

	if(numThreads > size())
	{
		{
			std::unique_lock<std::shared_mutex> lock(mainMutex_);
			std::unique_lock<std::shared_mutex> stopFlagsMutex(stopFlagsMutex_);

			workers_.reserve(numThreads);
			stopFlags_.reserve(numThreads);
		}

		for(size_t threadNum = workers_.size(); threadNum < numThreads; threadNum++)
		{
			stopFlags_.emplace_back(false);

			workers_.emplace_back([threadNum, this]() { _spawn(threadNum); });
		}
	}
}

} // namespace utilities
