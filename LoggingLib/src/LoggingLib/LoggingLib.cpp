#include <LoggingLib/LoggingLib.h>
#include <iostream>

void LOG_INFO(const char* preamble, const char* content)
{
	std::cout << "\033[34m"
			  << "[---INFO---]"
			  << "[" << preamble << "]: " << content << "\033[0m\n";
}

void LOG_INFO(const char* content)
{
	LOG_INFO("UNNAMED", content);
}

void LOG_WARN(const char* preamble, const char* content)
{
	std::cout << "\033[1;33m"
			  << "[---WARN---]"
			  << "[" << preamble << "]: " << content << "\033[0m\n";
}

void LOG_WARN(const char* content)
{
	LOG_WARN("UNNAMED", content);
}

void LOG_ERROR(const char* preamble, const char* content)
{
	std::cout << "\033[1;31m"
			  << "[---ERROR--]"
			  << "[" << preamble << "]: " << content << "\033[0m\n";

	throw std::runtime_error(content);
}

void LOG_ERROR(const char* content)
{
	LOG_ERROR("UNNAMED", content);
}