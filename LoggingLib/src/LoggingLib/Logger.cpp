// __Related headers__
#include <LoggingLib/Logger.h>

// __C++ standard headers__
#include <mutex>

// __External headers__
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
	std::lock_guard lock(streamingMutex_);

	defaultStream_ = stream;
}

void Logger::setNamedChannelStream(const std::string& name, std::ostream& stream)
{
	std::lock_guard lock(streamingMutex_);

	namedStreamsMap_.emplace(name, stream);
}

void Logger::logOnChannel(LogType logType, const char* channelName, const char* logContent)
{
	{
		std::lock_guard lock(streamingMutex_);

		std::ostream& chosenStream =
			namedStreamsMap_.contains(channelName) ? namedStreamsMap_.at(channelName).get() : defaultStream_.get();

		const auto frame = colorfulFramesMap.at(logType);

		chosenStream.write(frame, static_cast<int64_t>(std::strlen(frame)));

		const auto preamble = preamblesMap.at(logType);

		chosenStream.write(preamble, static_cast<int64_t>(std::strlen(preamble)));

		chosenStream.write(fmt::format("[{}] ", channelName).c_str(), static_cast<int64_t>(std::strlen(channelName) + 3));

		chosenStream.write(logContent, static_cast<int64_t>(std::strlen(logContent)));

		chosenStream.write("\033[0m\n", 4);

		chosenStream.flush();
	}

	if(logType == LogType::ERROR)
	{
		throw std::runtime_error(logContent);
	}
}

void Logger::resetLogger()
{
	defaultStream_ = std::cout;

	namedStreamsMap_.clear();
}

} // namespace loggingLib