/**********************
 * Test suite for 'ai_projects'
 *
 * Copyright (c) 2023
 *
 * by Wiktor Prosowicz
 **********************/

#include <stdexcept>
#include <string>
#include <strstream>
#include <vector>

#include <LoggingLib/LoggingLib.hpp>
#include <fmt/core.h>
#include <gtest/gtest.h>

namespace
{
/*****************************
 *
 * Test Fixture
 *
 *****************************/

class TestLoggingLib : public testing::Test
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

TEST_F(TestLoggingLib, testDefaultStreamLogging)
{
	std::ostringstream defaultStream;

	LOG_RESET_LOGGER()
	LOG_SET_DEFAULT_STREAM(defaultStream)

	LOG_WARN("Channel 1", "Message number 1")
	LOG_INFO("Channel 1", "Message number 2")

	EXPECT_THROW(LOG_ERROR("Channel 1", "Message number 3"), std::runtime_error);

	LOG_WARN("Channel 2", "Message number 4")
	LOG_INFO("Channel 2", "Message number 5")

	EXPECT_THROW(LOG_ERROR("Channel 2", "Message number 6"), std::runtime_error);

	const std::vector<std::string> expectedLogs{"\033[1;33m[ WARN][Channel 1] Message number 1\033[0m",
												"\033[34m[ INFO][Channel 1] Message number 2\033[0m",
												"\033[1;31m[ERROR][Channel 1] Message number 3\033[0m",
												"\033[1;33m[ WARN][Channel 2] Message number 4\033[0m",
												"\033[34m[ INFO][Channel 2] Message number 5\033[0m",
												"\033[1;31m[ERROR][Channel 2] Message number 6\033[0m"};

	checkHarvestedLogs(defaultStream.str(), expectedLogs);
}

TEST_F(TestLoggingLib, testNamedChannelsLogging)
{
	std::ostringstream defaultStream;

	std::ostringstream firstChannelStream;
	std::ostringstream secondChannelStream;

	LOG_RESET_LOGGER();
	LOG_SET_DEFAULT_STREAM(defaultStream);

	LOG_SET_NAMED_STREAM("Channel 1", firstChannelStream);
	LOG_SET_NAMED_STREAM("Channel 2", secondChannelStream);

	LOG_INFO("Unnamed", "Message 1");
	LOG_INFO("Channel 1", "Message 2");
	LOG_INFO("Channel 1", "Message 3");
	LOG_INFO("Unnamed", "Message 4");
	LOG_INFO("Channel 2", "Message 5");
	LOG_INFO("Channel 1", "Message 6");
	LOG_INFO("Channel 2", "Message 7");

	LOG_SET_NAMED_STREAM("Channel 2", defaultStream);

	LOG_INFO("Channel 2", "Message 8");

	checkHarvestedLogs(defaultStream.str(),
					   {"\033[34m[ INFO][Unnamed] Message 1\033[0m",
						"\033[34m[ INFO][Unnamed] Message 4\033[0m",
						"\033[34m[ INFO][Channel 2] Message 8\033[0m"});

	checkHarvestedLogs(firstChannelStream.str(),
					   {"\033[34m[ INFO][Channel 1] Message 2\033[0m",
						"\033[34m[ INFO][Channel 1] Message 3\033[0m",
						"\033[34m[ INFO][Channel 1] Message 6\033[0m"});

	checkHarvestedLogs(
		secondChannelStream.str(),
		{"\033[34m[ INFO][Channel 2] Message 5\033[0m", "\033[34m[ INFO][Channel 2] Message 7\033[0m"});
}

} // namespace
