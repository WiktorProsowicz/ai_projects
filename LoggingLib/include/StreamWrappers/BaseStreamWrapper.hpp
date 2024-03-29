#ifndef LOGGINGLIB_INCLUDE_STREAMWRAPPERS_BASESTREAMWRAPPER_HPP
#define LOGGINGLIB_INCLUDE_STREAMWRAPPERS_BASESTREAMWRAPPER_HPP

// __C++ standard headers__
#include <ostream>

// __Own software headers__
#include <StreamWrappers/IStreamWrapper.hpp>

namespace streamWrappers
{

/**
 * @brief Class providing template streaming algorithm that operates on wrapped std::ostream.
 * 
 */
class BaseStreamWrapper : public IStreamWrapper
{
public:
	/**
	 * @brief Creates a new stack of `stream` < BaseStreamWrapper < `WrapperType`. 
	 * 
	 * @tparam WrapperType Type of the most external wrapper.
	 * @param stream std::ostream to be wrapped by the intermediate BaseStreamWrapper.
	 * @return IStreamWrapper Pointer to the external wrapper.
	 */
	template <typename WrapperType>
	static IStreamWrapperPtr spawnWrapped(std::ostream& stream) requires std::is_base_of_v<IStreamWrapper, WrapperType>
	{
		return std::make_shared<WrapperType>(std::make_shared<BaseStreamWrapper>(stream));
	}

	BaseStreamWrapper() = delete; /// Default constructor.

	/**
     * @brief Creates a new BaseStreamWrapper and assigns an std::ostream to which the content shall be streamed.
     * 
     * @param stream 
     */
	explicit BaseStreamWrapper(std::ostream& stream)
		: stream_(stream)
	{ }

	~BaseStreamWrapper() override = default; /// Default virtual destructor.

	void putCharString(const char* charString) override
	{
		put(charString);
	}

	/**
     * @brief Streams the given `content` to the referenced std::ostream.
     * 
     * @param content Content to stream into the underlying std::ostream.
     */
	template <typename StreamedType>
	void put(StreamedType&& content)
	{
		stream_ << content;

		stream_.flush();
	}

	/**
	 * @brief Returns a reference to the underlying stream.
	 * 
	 * @return std::ostream& 
	 */
	std::ostream& getStream()
	{
		return stream_;
	}

private:
	std::ostream& stream_;
};

} // namespace streamWrappers

#endif