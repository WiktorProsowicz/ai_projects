#ifndef LOGGINGLIB_INCLUDE_LOGGINGLIB_LOGGER_H
#define LOGGINGLIB_INCLUDE_LOGGINGLIB_LOGGER_H

// __CPP headers__
#include <iostream>
#include <functional>
#include <map>
#include <mutex>

// __Own software headers__
#include <StreamWrappers/BaseStreamWrapper.hpp>

namespace loggingLib
{
enum class LogType : uint8_t
{
	WARN,
	ERROR,
	INFO
};

/**
 * @brief Singleton class used to stream logs to the desired type of output stream. Logger operates on named logging channels and can assign types of logging
 * to emphasise the concrete message and apply coloring.
 *
 */
class Logger
{
public:
	Logger(const Logger&) = delete;			   // Copy constructor
	Logger(Logger&&) = delete;				   // Move constructor
	Logger& operator=(const Logger&) = delete; // Copy assignment
	Logger& operator=(Logger&&) = delete;	   // Move assignment

	/**
     * @brief Returns the global logger instance.
     *
     * @return Global logger.
     */
	static Logger& getInstance();

	/**
	 * @brief Streams the given info message in on the channel specified by `channelName`.
	 *
	 * @param channelName Name of the channel to log on.
	 * @param logContent Message content,
	 */
	void logInfoOnChannel(const char* channelName, const char* logContent);

	/**
	 * @brief Streams the given warning message in on the channel specified by `channelName`.
	 *
	 * @param channelName Name of the channel to log on.
	 * @param logContent Message content,
	 */
	void logWarnOnChannel(const char* channelName, const char* logContent);

	/**
	 * @brief Streams the given error message in on the channel specified by `channelName`.
	 * Also this kind of log throws an std::runtime_error and, if not caught, terminates the program.
	 *
	 * @param channelName Name of the channel to log on.
	 * @param logContent Message content,
	 */
	void logErrorOnChannel(const char* channelName, const char* logContent);

	/**
	 * @brief Sets the default logging stream.
	 *
	 * @param stream Stream to be set to default.
	 */
	void setDefaultStream(std::ostream& stream);

	/// @overload
	void setDefaultStream(streamWrappers::IStreamWrapperPtr stream);

	/**
	 * @brief Sets the stream associated to the name of a specific channel.
	 *
	 * @param name Name of the channel to set the stream to.
	 * @param stream Stream to set to the channel.
	 */
	void setNamedChannelStream(const std::string& name, std::ostream& stream);

	/// @overload
	void setNamedChannelStream(const std::string& name, streamWrappers::IStreamWrapperPtr stream);

	/**
	 * @brief Cleans the internal logger's configuration, i.e. streams associated to channels, default stream etc.
	 *
	 */
	void reset();

private:
	/**
     * @brief Constructs a global logger object.
     *
     * @param stream Initial stream that logger shall send logs to.
     *
     */
	Logger()
		: defaultStream_(std::make_shared<streamWrappers::BaseStreamWrapper>(std::cout))
		, namedStreamsMap_()
		, streamingMutex_()
	{ }

	/**
     * @brief Streams the given message into the channel specified by the `channelName`.
	 * Type of the log depends on the given type argument.
	 * The message is streamed to the std::ostream& stored in the channel_name-stream map and, if not specified, on the default stream.
     *
     * @param logType Type of the streamed log.
     * @param channelName Name of the log channel.
     * @param logContent Message content.
     */
	void logOnChannel(LogType logType, const char* channelName, const char* logContent);

private:
	streamWrappers::IStreamWrapperPtr defaultStream_;
	std::map<std::string, streamWrappers::IStreamWrapperPtr> namedStreamsMap_;
	std::mutex streamingMutex_;

	const static inline std::map<LogType, const char*> colorfulFramesMap{
		{LogType::INFO, "\033[34m"}, {LogType::WARN, "\033[1;33m"}, {LogType::ERROR, "\033[1;31m"}};

	const static inline std::map<LogType, const char*> preamblesMap{
		{LogType::INFO, "[ INFO]"}, {LogType::WARN, "[ WARN]"}, {LogType::ERROR, "[ERROR]"}};
};
} // namespace loggingLib

#endif
