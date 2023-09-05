#ifndef LOGGINGLIB_INCLUDE_STREAMWRAPPERS_BASESTREAMWRAPPER_HPP
#define LOGGINGLIB_INCLUDE_STREAMWRAPPERS_BASESTREAMWRAPPER_HPP

// __C++ standard headers__
#include <memory>
#include <ostream>

namespace streamWrappers
{

/**
 * @brief Class providing template streaming algorithm that operates on wrapped std::ostream.
 * 
 */
class BaseStreamWrapper
{
public:
	BaseStreamWrapper() = delete; /// Default constructor.

	/**
     * @brief Creates a new BaseStreamWrapper and assigns an std::ostream to which the content shall be streamed.
     * 
     * @param stream 
     */
	explicit BaseStreamWrapper(std::ostream& stream)
		: stream_(stream)
	{ }

	virtual ~BaseStreamWrapper() = default; /// Default virtual destructor.

	/**
     * @brief Streams the given `content` to the referenced std::ostream.
     * Additionally performs two actions, both before and after streaming, whose effect depends on the class inheriting from the BaseStreamWrapper.
     * The streamed content is changed in a way specified by the base/inheriting class.
     * 
     * @param content Content to stream into the underlying std::ostream.
     */
	template <typename StreamedType>
	void put(StreamedType&& content)
	{
		_beforeStream();

		if constexpr(std::is_convertible_v<StreamedType, const char*>)
		{
			stream_ << _modifyCharInput(content);
		}
		else
		{
			stream_ << content;
		}

		_afterStream();
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

protected:
	/**
     * @brief Performs a certain action before streaming any content to the underlying std::ostream.
     * 
     */
	virtual void _beforeStream() { }

	/**
     * @brief Performs a certain action after streaming any content to the underlying std::ostream.
     * 
     */
	virtual void _afterStream() { }

	/**
     * @brief Modifies the given input in a way specified by the class following the BaseStreamWrapper's interface.
     * 
     * @param input Content to be modify.
     * @return StreamType&& Modified version of the input.
     */
	virtual std::string _modifyCharInput(const char* input)
	{
		return input;
	}

private:
	std::ostream& stream_;
};

using BaseStreamWrapperPtr = std::shared_ptr<BaseStreamWrapper>;

} // namespace streamWrappers

#endif