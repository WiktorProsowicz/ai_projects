#ifndef MLCORE_INCLUDE_MLCORE_TENSORINITIALIZERS_RANGETENSORINITIALIZER_HPP
#define MLCORE_INCLUDE_MLCORE_TENSORINITIALIZERS_RANGETENSORINITIALIZER_HPP

#include <limits>
#include <stdexcept>

#include "MLCore/TensorInitializers/ITensorInitializer.hpp"

namespace mlCore::tensorInitializers
{
/**
 * @brief Yields values from linear range
 *
 */
template <class ValueType>
class RangeTensorInitializer : public ITensorInitializer<ValueType>
{
public:
	/**
	 * @brief Construct a new Range Tensor Initializer object
	 *
	 * @param firstValue Initial value to be returned and appended with each yield
	 * @param step The increment factor of the initializer's value
	 * @param maxValue The border value at which the initializer stops
	 */
	explicit RangeTensorInitializer(ValueType firstValue,
									ValueType step = 1,
									ValueType maxValue = std::numeric_limits<ValueType>::max())
		: _currentValue(firstValue)
		, _maxValue(maxValue)
		, _step(step)

	{}

	RangeTensorInitializer(RangeTensorInitializer&&) = delete;				   // Move ctor
	RangeTensorInitializer(const RangeTensorInitializer&) = delete;			   // Copy ctor
	RangeTensorInitializer& operator=(RangeTensorInitializer&&) = delete;	   // Move assign
	RangeTensorInitializer& operator=(const RangeTensorInitializer&) = delete; // Copy assign

	~RangeTensorInitializer() override = default;

	ValueType yield() const override
	{
		if(!canYield())
		{
			throw std::out_of_range("Cannot obtain value from RangeTensorYielder.");
		}

		ValueType out = _currentValue;
		_currentValue += _step;
		return out;
	}

	bool canYield() const override
	{
		return _currentValue <= _maxValue;
	}

private:
	mutable ValueType _currentValue;
	ValueType _maxValue;
	ValueType _step;
};
} // namespace mlCore::tensorInitializers

#endif
