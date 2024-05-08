#ifndef MLCORE_INCLUDE_MODELS_ICALLBACK_HPP
#define MLCORE_INCLUDE_MODELS_ICALLBACK_HPP

#include <memory>
#include <type_traits>

namespace interfaces
{

/**
 * @brief Specifies the moment at which the callback should be called.
 *
 */
enum class CallbackMode : uint8_t
{
	START_OF_BATCH = 0x01 << 1,
	END_OF_BATCH = 0x01 << 2,
	END_OF_TRAINING = 0x01 << 3,
	START_OF_TRAINING = 0x01 << 4,
	AFTER_GRADIENTS_UPDATE = 0x01 << 5
};

/**
 * @brief Interface for classes executing specific actions at specific points in time.
 *
 */
class Callback
{
	using CMType = std::underlying_type_t<CallbackMode>;

public:
	Callback() = default;

	Callback(const Callback&) = default;
	Callback(Callback&&) = default;
	Callback& operator=(const Callback&) = default;
	Callback& operator=(Callback&&) = default;

	virtual ~Callback() = default;

	/**
	 * @brief Tells the callback to execute its task.
	 *
	 */
	virtual void call() = 0;

	/// Adds single CallbackMode to overall callback's mode.
	void addMode(const CallbackMode mode)
	{
		_mode |= static_cast<CMType>(mode);
	}

	/// Removes single CallbackMode from the overall callback's mode.
	void removeMode(const CallbackMode mode)
	{
		_mode &= (~static_cast<CMType>(mode));
	}

	/// Tells if the overall callback's mode has given mode.
	bool hasMode(const CallbackMode mode) const
	{
		return (_mode & static_cast<CMType>(mode)) != 0;
	}

private:
	CMType _mode = 0;
};

using CallbackPtr = std::shared_ptr<Callback>;
} // namespace interfaces

#endif
