#ifndef LOGGINGLIB_INCLUDE_STREAMWRAPPERS_DECOLORIZINGSTREAM_HPP
#define LOGGINGLIB_INCLUDE_STREAMWRAPPERS_DECOLORIZINGSTREAM_HPP

// __Related headers__
#include <StreamWrappers/BaseStreamWrapper.hpp>

// __C++ standard headers__
#include <regex>
#include <cstring>
#include <fstream>

namespace streamWrappers
{
/**
 * @brief Class deleting all of color-controlling characters from the streamed content.
 * 
 */
class DecolorizingStream : public BaseStreamWrapper
{
public:
	/**
     * @brief Creates a new DecolorizingStream.
     * 
     * @param stream Stream to pass to the base stream wrapper.
     */
	explicit DecolorizingStream(std::ostream& stream)
		: BaseStreamWrapper(stream)
	{ }

	~DecolorizingStream() override = default; /// Default destructor.

protected:
	std::string _modifyCharInput(const char* input) override
	{
		return std::regex_replace(input, coloringRegex_, "");
	}

private:
	static inline const std::regex coloringRegex_{"\\\033\\[\\d+;?\\d*m"};
};
} // namespace streamWrappers

#endif