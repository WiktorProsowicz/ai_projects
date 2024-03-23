#ifndef LOGGINGLIB_INCLUDE_LOGGINGLIB_STREAMWRAPPERS_ISTREAMWRAPPER_HPP
#define LOGGINGLIB_INCLUDE_LOGGINGLIB_STREAMWRAPPERS_ISTREAMWRAPPER_HPP

// __C++ standard headers__
#include <memory>

namespace streamWrappers
{
/**
 * @brief Interface for stream wrappers classes.
 * All stream wrappers are meant to follow decorator pattern, which enables applying certain modifications on the streamed content.
 *
 */
class IStreamWrapper
{
public:
	virtual ~IStreamWrapper() = default; // Virtual destructor.

	/**
     * @brief Streams given `charString` into the wrapped stream.
     *
     * @param charString Content to stream.
     */
	virtual void putCharString(const char* charString) = 0;
};

using IStreamWrapperPtr = std::shared_ptr<IStreamWrapper>;
} // namespace streamWrappers

#endif
