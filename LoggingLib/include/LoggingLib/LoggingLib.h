#ifndef LOGGINGLIB_LOGGINGLIB_H
#define LOGGINGLIB_LOGGINGLIB_H

/**
 * @brief log message having some informative content
 * 
 */
void LOG_INFO(const char* preamble, const char* content);
void LOG_INFO(const char* content);

/**
 * @brief log message that warns about something
 * 
 */
void LOG_WARN(const char* preamble, const char* content);
void LOG_WARN(const char* content);

/**
 * @brief log message and stop program with runtime exception
 * 
 */
void LOG_ERROR(const char* preamble, const char* content);
void LOG_ERROR(const char* content);

#endif