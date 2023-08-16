#include <Utilities/ThreadPool.h>

#include <functional>

namespace utilities
{
void ThreadPool::init(size_t numThreads)
{
	std::call_once(once_, [this, &numThreads] {
		std::unique_lock<std::shared_mutex> lock(mainMutex_);

		workers_.reserve(numThreads);

		for(size_t threadNum = 0; threadNum < numThreads; threadNum++)
		{
			workers_.emplace_back(std::bind(&ThreadPool::_spawn, this));
		}

		initted_ = true;
	});
}

void ThreadPool::terminate()
{
	{
		std::unique_lock<std::shared_mutex> lock(mainMutex_);
		if(_isRunning())
		{
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
}

void ThreadPool::_spawn()
{
	while(true)
	{
		bool obtainedATask = false;
		std::function<void()> runTask;

		{
			std::unique_lock<std::shared_mutex> lock(mainMutex_);
			condition_.wait(lock, [this, &obtainedATask, &runTask] {
				obtainedATask = tasks_.tryPop(runTask);
				return cancelled_ || stopped_ || obtainedATask;
			});
		}
		if(cancelled_ || (stopped_ && !obtainedATask))
		{
			return;
		}
		runTask();
	}
}

} // namespace utilities