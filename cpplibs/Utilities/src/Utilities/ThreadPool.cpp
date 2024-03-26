#include "Utilities/ThreadPool.h"

#include <functional>

namespace utilities
{
void ThreadPool::init(size_t numThreads)
{
	std::call_once(_once,
				   [this, &numThreads]
				   {
					   _initted = true;

					   resize(numThreads);
				   });
}

void ThreadPool::terminate()
{
	{
		if(_isRunning())
		{
			std::unique_lock<std::shared_mutex> lock(_flagsMutex);
			_stopped = true;
		}
		else
		{
			return;
		}
	}

	_condition.notify_all();

	for(auto& worker : _workers)
	{
		worker.join();
	}
}

void ThreadPool::cancel()
{
	{
		std::unique_lock<std::shared_mutex> lock(_mainMutex);
		if(_isRunning())
		{
			_cancelled = true;
		}
		else
		{
			return;
		}
	}

	_tasks.clear();
	_condition.notify_all();

	for(auto& worker : _workers)
	{
		worker.join();
	}

	_workers.clear();
}

void ThreadPool::_spawn(size_t workerId)
{
	while(true)
	{
		bool obtainedATask = false;
		std::function<void()> runTask;

		{
			std::unique_lock<std::shared_mutex> lock(_mainMutex);

			_condition.wait(lock,
							[this, &obtainedATask, &runTask, &workerId]
							{
								obtainedATask = _tasks.tryPop(runTask);

								std::shared_lock<std::shared_mutex> flagsLock(_flagsMutex);
								std::shared_lock<std::shared_mutex> stopFlagsLock(_stopFlagsMutex);

								return _cancelled || _stopped || obtainedATask || _stopFlags.at(workerId);
							});
		}

		{
			std::shared_lock<std::shared_mutex> flagsLock(_flagsMutex);

			// In case the whole pool has been cancelled - don't care about the following task
			// In case the pool has been stopped - run tasks until there are any
			if(_cancelled || (_stopped && !obtainedATask))
			{
				return;
			}
		}

		if(obtainedATask)
		{
			runTask();
		}

		{
			std::shared_lock<std::shared_mutex> stopFlagsLock(_stopFlagsMutex);

			// Closing the individual thread only after possible processing of the task
			if(_stopFlags.at(workerId))
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
			std::unique_lock<std::shared_mutex> stopFlagsMutex(_stopFlagsMutex);

			for(size_t threadNum = numThreads; threadNum < _workers.size(); threadNum++)
			{
				_stopFlags.at(threadNum) = true;
			}
		}

		_condition.notify_all();

		for(size_t threadNum = numThreads; threadNum < _workers.size(); threadNum++)
		{
			_workers.at(threadNum).join();
		}

		_workers.resize(numThreads);
		_stopFlags.resize(numThreads);

		return;
	}

	if(numThreads > size())
	{
		{
			std::unique_lock<std::shared_mutex> lock(_mainMutex);
			std::unique_lock<std::shared_mutex> stopFlagsMutex(_stopFlagsMutex);

			_workers.reserve(numThreads);
			_stopFlags.reserve(numThreads);
		}

		for(size_t threadNum = _workers.size(); threadNum < numThreads; threadNum++)
		{
			_stopFlags.emplace_back(false);

			_workers.emplace_back([threadNum, this]() { _spawn(threadNum); });
		}
	}
}

} // namespace utilities
