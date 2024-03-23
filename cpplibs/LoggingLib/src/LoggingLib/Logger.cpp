#include "LoggingLib/Logger.h"

#include <mutex>

#include <fmt/format.h>

namespace loggingLib
{
Logger& Logger::getInstance()
{
	static Logger globalLogger;

	return globalLogger;
}

void Logger::logInfoOnChannel(const char* channelName, const char* logContent)
{
	logOnChannel(LogType::INFO, channelName, logContent);
}

void Logger::logWarnOnChannel(const char* channelName, const char* logContent)
{
	logOnChannel(LogType::WARN, channelName, logContent);
}

void Logger::logErrorOnChannel(const char* channelName, const char* logContent)
{
	logOnChannel(LogType::ERROR, channelName, logContent);
}

void Logger::setDefaultStream(std::ostream& stream)
{
	setDefaultStream(std::make_shared<streamWrappers::BaseStreamWrapper>(stream));
}

void Logger::setDefaultStream(const streamWrappers::IStreamWrapperPtr stream)
{
	std::lock_guard lock(streamingMutex_);

	defaultStream_ = stream;
}

void Logger::setNamedChannelStream(const std::string& name, std::ostream& stream)
{
	setNamedChannelStream(name, std::make_shared<streamWrappers::BaseStreamWrapper>(stream));
}

void Logger::setNamedChannelStream(const std::string& name, streamWrappers::IStreamWrapperPtr stream)
{
	std::lock_guard lock(streamingMutex_);

	if(namedStreamsMap_.contains(name))
	{
		namedStreamsMap_.at(name) = stream;
	}
	else
	{
		namedStreamsMap_.emplace(name, stream);
	}
}

void Logger::logOnChannel(LogType logType, const char* channelName, const char* logContent)
{
	{
		std::lock_guard lock(streamingMutex_);

		streamWrappers::IStreamWrapperPtr chosenStream =
			namedStreamsMap_.contains(channelName) ? namedStreamsMap_.at(channelName) : defaultStream_;

		const auto* frame = colorfulFramesMap.at(logType);

		chosenStream->putCharString(frame);

		const auto* preamble = preamblesMap.at(logType);

		chosenStream->putCharString(preamble);

		chosenStream->putCharString(fmt::format("[{}] ", channelName).c_str());

		chosenStream->putCharString(logContent);

		chosenStream->putCharString("\033[0m\n");
	}

	if(logType == LogType::ERROR)
	{
		throw std::runtime_error(logContent);
	}
}

void Logger::reset()
{
	setDefaultStream(std::cout);

	namedStreamsMap_.clear();
}

} // namespace loggingLib
