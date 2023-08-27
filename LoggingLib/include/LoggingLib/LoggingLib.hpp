#ifndef LOGGINGLIB_INCLUDE_LOGGINGLIB_LOGGINGLIB_HPP
#define LOGGINGLIB_INCLUDE_LOGGINGLIB_LOGGINGLIB_HPP

// __C++ standard headers__
#include <sstream>

// __Own headers__
#include <LoggingLib/Logger.h>

/**
 * @brief Logs a message having some informative content.
 * 
 */
#define LOG_INFO(preamble, content)                                                                                              \
	loggingLib::Logger::getInstance().logInfoOnChannel(preamble, (std::stringstream{} << content).str().c_str())

/**
 * @brief Log a message that warns about something.
 * 
 */
#define LOG_WARN(preamble, content)                                                                                              \
	loggingLib::Logger::getInstance().logWarnOnChannel(preamble, (std::stringstream{} << content).str().c_str())

/**
 * @brief Logs a message and stop program with runtime exception.
 * 
 */
#define LOG_ERROR(preamble, content)                                                                                             \
	loggingLib::Logger::getInstance().logErrorOnChannel(preamble, (std::stringstream{} << content).str().c_str())

#endif