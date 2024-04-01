#ifndef LOGGINGLIB_INCLUDE_STREAMWRAPPERS_DECOLORIZINGSTREAM_HPP
#define LOGGINGLIB_INCLUDE_STREAMWRAPPERS_DECOLORIZINGSTREAM_HPP

#include <cstring>
#include <fstream>
#include <regex>

#include "StreamWrappers/IStreamWrapper.hpp"

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
		: _wrappedStream(std::move(wrappedStream))
	{}

	DecolorizingStream(const DecolorizingStream&) = default;
	DecolorizingStream(DecolorizingStream&&) = default;
	DecolorizingStream& operator=(const DecolorizingStream&) = default;
	DecolorizingStream& operator=(DecolorizingStream&&) = default;

	~DecolorizingStream() override = default; /// Default destructor.

	void putCharString(const char* charString) override
	{
		_wrappedStream->putCharString(std::regex_replace(charString, _coloringRegex, "").c_str());
	}

private:
	static inline const std::regex _coloringRegex{"\\\033\\[\\d+;?\\d*m"};

	IStreamWrapperPtr _wrappedStream;
};
} // namespace streamWrappers

#endif
