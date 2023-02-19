#ifndef LOGGINGLIB_LOGGINGLIB_H
#define LOGGINGLIB_LOGGINGLIB_H

#include <string>

/**
 * @brief log message having some informative content
 * 
 */
void LOG_INFO(const std::string& preamble, const std::string& content);
void LOG_INFO(const std::string& content);

/**
 * @brief log message that warns about something
 * 
 */
void LOG_WARN(const std::string& preamble, const std::string& content);
void LOG_WARN(const std::string& content);

/**
 * @brief log message and stop program with runtime exception
 * 
 */
void LOG_ERROR(const std::string& preamble, const std::string& content);
void LOG_ERROR(const std::string& content);

#endif