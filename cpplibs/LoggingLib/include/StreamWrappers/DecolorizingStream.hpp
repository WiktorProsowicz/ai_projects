#ifndef LOGGINGLIB_INCLUDE_STREAMWRAPPERS_DECOLORIZINGSTREAM_HPP
#define LOGGINGLIB_INCLUDE_STREAMWRAPPERS_DECOLORIZINGSTREAM_HPP

// __C++ standard headers__
#include <regex>
#include <cstring>
#include <fstream>

// __Own software headers__
#include <StreamWrappers/IStreamWrapper.hpp>

namespace streamWrappers
{
/**
 * @brief Class deleting all of color-controlling characters from the streamed content.
 *
 */
class DecolorizingStream : public IStreamWrapper
{
public:
	/**
     * @brief Creates a new DecolorizingStream.
     *
     * @param stream Stream to pass to the base stream wrapper.
     */
	explicit DecolorizingStream(IStreamWrapperPtr wrappedStream)
		: wrappedStream_(wrappedStream)
	{ }

	~DecolorizingStream() override = default; /// Default destructor.

	void putCharString(const char* charString) override
	{
		wrappedStream_->putCharString(std::regex_replace(charString, coloringRegex_, "").c_str());
	}

private:
	static inline const std::regex coloringRegex_{"\\\033\\[\\d+;?\\d*m"};

	IStreamWrapperPtr wrappedStream_;
};
} // namespace streamWrappers

#endif
