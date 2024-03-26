/**********************
 * Test suite for 'ai_projects'
 *
 * Copyright (c) 2023
 *
 * by Wiktor Prosowicz
 **********************/

#include <memory>
#include <string>
#include <strstream>
#include <vector>

#include <StreamWrappers/BaseStreamWrapper.hpp>
#include <StreamWrappers/DecolorizingStream.hpp>
#include <fmt/core.h>
#include <gtest/gtest.h>

namespace
{
/*****************************
 *
 * Test Fixture
 *
 *****************************/

class TestStreamWrappers : public testing::Test
{
protected:
	static void checkHarvestedLogs(const std::string& logsBulk, const std::vector<std::string>& expectedLogs)
	{
		std::istringstream logsStream(logsBulk.c_str());

		auto expectedLogsIt = expectedLogs.cbegin();
		std::string harvestedLog; // placeholder

		while(std::getline(logsStream, harvestedLog))
		{
			if(expectedLogsIt == expectedLogs.cend())
			{
				FAIL() << fmt::format("Harvested more logs than expected!");
			}

			EXPECT_STREQ(harvestedLog.c_str(), expectedLogsIt->c_str());

			expectedLogsIt++;
		}

		if(expectedLogsIt != expectedLogs.cend())
		{
			FAIL() << fmt::format("Harvested less logs than expected!");
		}
	}
};

/*****************************
 *
 * Particular test calls
 *
 *****************************/

TEST_F(TestStreamWrappers, testBaseStreamWrapper)
{
	std::stringstream strStream;

	auto baseWrapper = std::make_shared<streamWrappers::BaseStreamWrapper>(strStream);

	baseWrapper->put("Message 1\n");
	baseWrapper->put("Message 2\n");
	baseWrapper->put("Message 3\n");

	checkHarvestedLogs(strStream.str(), {"Message 1", "Message 2", "Message 3"});
}

TEST_F(TestStreamWrappers, testDecolorizingStream)
{
	std::stringstream strStream;

	auto decolorizingStream =
		streamWrappers::BaseStreamWrapper::spawnWrapped<streamWrappers::DecolorizingStream>(strStream);

	decolorizingStream->putCharString("\033[31;0m[ WARN][Unnamed] Message number 1\033[0m\n");
	decolorizingStream->putCharString("\033[32;1m[ INFO][Unnamed] Message number 2\033[0m\n");
	decolorizingStream->putCharString("\033[33m[ERROR][Unnamed] Message number 3\033[0m\n");

	checkHarvestedLogs(strStream.str(),
					   {"[ WARN][Unnamed] Message number 1",
						"[ INFO][Unnamed] Message number 2",
						"[ERROR][Unnamed] Message number 3"});
}

} // namespace
