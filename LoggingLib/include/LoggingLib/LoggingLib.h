#ifndef LOGGINGLIB_LOGGINGLIB_H
#define LOGGINGLIB_LOGGINGLIB_H

#include <string>

/**
 * @brief log message having some informative content
 * 
 */
#define LOG_INFO(preamble, content)                                                                \
	std::cout << "\033[34m"                                                                        \
			  << "[---INFO---]"                                                                    \
			  << "[" << preamble << "]: " << content << "\033[0m\n";

/**
 * @brief log message that warns about something
 * 
 */
#define LOG_WARN(preamble, content)                                                                \
	std::cout << "\033[1;33m"                                                                      \
			  << "[---WARN---]"                                                                    \
			  << "[" << preamble << "]: " << content << "\033[0m\n";

/**
 * @brief log message and stop program with runtime exception
 * 
 */
#define LOG_ERROR(preamble, content)                                                               \
	std::cout << "\033[1;31m"                                                                      \
			  << "[---ERROR--]"                                                                    \
			  << "[" << preamble << "]: " << content << "\033[0m\n";                               \
	throw std::runtime_error(content);

#endif