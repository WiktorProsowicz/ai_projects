#include <LoggingLib/LoggingLib.h>
#include <iostream>

void LOG_INFO(const std::string& preamble, const std::string& content)
{
	std::cout << "\033[34m"
			  << "[---INFO---]"
			  << "[" << preamble << "]: " << content << "\033[0m\n";
}

void LOG_INFO(const std::string& content)
{
	LOG_INFO("UNNAMED", content);
}

void LOG_WARN(const std::string& preamble, const std::string& content)
{
	std::cout << "\033[1;33m"
			  << "[---WARN---]"
			  << "[" << preamble << "]: " << content << "\033[0m\n";
}

void LOG_WARN(const std::string& content)
{
	LOG_WARN("UNNAMED", content);
}

void LOG_ERROR(const std::string& preamble, const std::string& content)
{
	std::cout << "\033[1;31m"
			  << "[---ERROR--]"
			  << "[" << preamble << "]: " << content << "\033[0m\n";

	throw std::runtime_error(content);
}

void LOG_ERROR(const std::string& content)
{
	LOG_ERROR("UNNAMED", content);
}