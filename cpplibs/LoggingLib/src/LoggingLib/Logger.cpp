#include "LoggingLib/Logger.h"

#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>

#include <fmt/core.h>

#include "StreamWrappers/BaseStreamWrapper.hpp"

namespace loggingLib
{
Logger& Logger::getInstance()
{
	static Logger globalLogger;

	return globalLogger;
}

void Logger::logInfoOnChannel(const char* channelName, const char* logContent)
{
	_logOnChannel(LogType::INFO, channelName, logContent);
}

void Logger::logWarnOnChannel(const char* channelName, const char* logContent)
{
	_logOnChannel(LogType::WARN, channelName, logContent);
}

void Logger::logErrorOnChannel(const char* channelName, const char* logContent)
{
	_logOnChannel(LogType::ERROR, channelName, logContent);
}

void Logger::setDefaultStream(std::ostream& stream)
{
	setDefaultStream(std::make_shared<streamWrappers::BaseStreamWrapper>(stream));
}

void Logger::setDefaultStream(const streamWrappers::IStreamWrapperPtr& stream)
{
	const std::lock_guard lock(_streamingMutex);

	_defaultStream = stream;
}

void Logger::setNamedChannelStream(const std::string& name, std::ostream& stream)
{
	setNamedChannelStream(name, std::make_shared<streamWrappers::BaseStreamWrapper>(stream));
}

void Logger::setNamedChannelStream(const std::string& name, streamWrappers::IStreamWrapperPtr stream)
{
	const std::lock_guard lock(_streamingMutex);

	if(_namedStreamsMap.contains(name))
	{
		_namedStreamsMap.at(name) = stream;
	}
	else
	{
		_namedStreamsMap.emplace(name, stream);
	}
}

void Logger::_logOnChannel(const LogType logType, const char* channelName, const char* logContent)
{
	{
		const std::lock_guard lock(_streamingMutex);

		const streamWrappers::IStreamWrapperPtr chosenStream =
			_namedStreamsMap.contains(channelName) ? _namedStreamsMap.at(channelName) : _defaultStream;

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

	_namedStreamsMap.clear();
}

} // namespace loggingLib
