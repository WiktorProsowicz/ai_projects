/**********************
 * Test suite for 'ai_projects'
 * 
 * Copyright (c) 2023
 * 
 * by Wiktor Prosowicz
 **********************/

// __Tested headers__
#include <Utilities/ThreadPool.h>

// __CPP headers__
#include <chrono>

// __External software__
#include <gtest/gtest.h>
#include <fmt/format.h>

// __Own software__
#include <LoggingLib/LoggingLib.hpp>

namespace
{
/*************************
 * 
 * Common data structures
 * 
 *************************/

class Task
{
public:
	using TimePoint = std::chrono::system_clock::time_point;

	Task() = delete;
	Task(size_t taskId, size_t waitingInterval = 1000)
		: waitingInterval_(waitingInterval)
		, id_(taskId)
	{ }

	void run()
	{
		startTime_ = std::chrono::system_clock::now();

		std::this_thread::sleep_for(std::chrono::milliseconds(waitingInterval_));

		endTime_ = std::chrono::system_clock::now();

		// LOG_INFO("TestThreadPool",
		// 		 fmt::format("Task number {} worked between {} and {}. ({}ms)",
		// 					 id_,
		// 					 std::chrono::duration_cast<std::chrono::milliseconds>(startTime_.time_since_epoch()).count(),
		// 					 std::chrono::duration_cast<std::chrono::milliseconds>(endTime_.time_since_epoch()).count(),
		// 					 std::chrono::duration_cast<std::chrono::milliseconds>(endTime_ - startTime_).count()));

		hasBeenRun_ = true;
	}

	TimePoint getStart() const
	{
		return startTime_;
	}

	TimePoint getEnd() const
	{
		return endTime_;
	}

	bool hasBeenRun() const
	{
		return hasBeenRun_;
	}

	size_t getId() const
	{
		return id_;
	}

private:
	size_t waitingInterval_;
	size_t id_;
	bool hasBeenRun_ = false;

	TimePoint startTime_{};
	TimePoint endTime_{};
};

/*****************************
 * 
 * Test Fixture
 * 
 *****************************/
class TestThreadPool : public testing::Test
{
protected:
	/**
	 * @brief Creates a vector of tasks.
	 * 
	 * @param numberOfTasks Number of tasks to create.
	 * @param waitingInterval Number of miliseconds each task need to wait when run.
	 * @return Created tasks.
	 */
	static std::vector<Task> createTasks(size_t numberOfTasks, size_t waitingInterval)
	{
		std::vector<Task> tasks;

		for(size_t taskNumber = 0; taskNumber < numberOfTasks; taskNumber++)
		{
			tasks.emplace_back(taskNumber, waitingInterval);
		}

		return tasks;
	}

	/**
	 * @brief Computes duration of the run tasks.
	 * 
	 * @param tasks Vector of tasks that have been run.
	 * @return Time span between the start of the earliest task and the end of the latest one.
	 */
	static auto getDuration(const std::vector<Task>& tasks)
	{
		auto minTime = std::min_element(tasks.cbegin(), tasks.cend(), [](const auto& task1, const auto& task2) {
			return task1.getStart().time_since_epoch() < task2.getStart().time_since_epoch();
		});

		auto maxTime = std::max_element(tasks.cbegin(), tasks.cend(), [](const auto& task1, const auto& task2) {
			return task1.getEnd().time_since_epoch() < task2.getEnd().time_since_epoch();
		});

		return std::chrono::duration_cast<std::chrono::milliseconds>(maxTime->getEnd() - minTime->getStart()).count();
	}
};
} // namespace

/*****************************
 * 
 * Particular test calls
 * 
 *****************************/

TEST_F(TestThreadPool, testTasksProcessing)
{
	std::vector<std::pair<size_t, size_t>> tasksParams{{5, 1000}, {10, 500}, {20, 250}};

	for(const auto& [nTasks, interval] : tasksParams)
	{

		auto synchronousTasks = createTasks(nTasks, interval);

		LOG_INFO("TestThreadPool", "Running synchronous tasks...");

		for(auto& task : synchronousTasks)
		{
			task.run();
		}

		auto asynchronousTasks = createTasks(nTasks, interval);

		{
			utilities::ThreadPool pool;
			pool.init(std::thread::hardware_concurrency() / 2);

			LOG_INFO("TestThreadPool", "Running asynchronous tasks...");

			for(auto& task : asynchronousTasks)
			{
				pool.addJob([&task]() { task.run(); });
			}

			pool.terminate();
		}

		LOG_INFO("TestThreadPool",
				 fmt::format("Asynchronous tasks took {}ms and synchronous {}ms\n",
							 getDuration(asynchronousTasks),
							 getDuration(synchronousTasks)));

		ASSERT_LT(getDuration(asynchronousTasks), getDuration(synchronousTasks))
			<< fmt::format("While comparing the run of the {} tasks, each {}ms long, synchronous tasks unexpectedly "
						   "took less time than asynchronous ones.",
						   nTasks,
						   interval);
	}
}

TEST_F(TestThreadPool, testResizing)
{
	utilities::ThreadPool pool;

	ASSERT_EQ(pool.size(), 0) << "Thread pool should have size() == 0 before initializing.";

	static constexpr size_t kNumTasks = 30;
	static constexpr size_t kInterval = 100;

	auto asynchronousTasks = createTasks(kNumTasks, kInterval);

	for(auto& task : asynchronousTasks)
	{
		pool.addJob([&task]() { task.run(); });
	}

	{

		size_t numThreadsAtInit = std::thread::hardware_concurrency() / 2;

		pool.init(numThreadsAtInit);

		ASSERT_EQ(pool.size(), numThreadsAtInit) << "Unexpected size of the thread pool.";

		numThreadsAtInit /= 2;

		pool.resize(numThreadsAtInit);

		ASSERT_EQ(pool.size(), numThreadsAtInit) << "Unexpected size of the thread pool.";

		pool.terminate();
	}

	for(const auto& task : asynchronousTasks)
	{
		EXPECT_TRUE(task.hasBeenRun()) << fmt::format("Task number {} has not been run.", task.getId());
	}
}
