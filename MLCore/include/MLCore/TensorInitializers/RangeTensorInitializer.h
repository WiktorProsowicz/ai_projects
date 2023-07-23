#ifndef MLCORE_INCLUDE_MLCORE_TENSORINITIALIZERS_RANGETENSORINITIALIZER_H
#define MLCORE_INCLUDE_MLCORE_TENSORINITIALIZERS_RANGETENSORINITIALIZER_H

#include <MLCore/TensorInitializers/ITensorInitializer.h>
#include <limits>

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
	RangeTensorInitializer(ValueType firstValue,
						   ValueType step = 1,
						   ValueType maxValue = std::numeric_limits<ValueType>::max());

	RangeTensorInitializer(RangeTensorInitializer&&) = delete; // Move ctor
	RangeTensorInitializer(const RangeTensorInitializer&) = delete; // Copy ctor
	RangeTensorInitializer& operator=(RangeTensorInitializer&&) = delete; // Move assign
	RangeTensorInitializer& operator=(const RangeTensorInitializer&) = delete; // Copy assign

	~RangeTensorInitializer() = default;

	ValueType yield() const override;

	bool canYield() const override;

private:
	mutable ValueType currentValue_;
	ValueType maxValue_;
	ValueType step_;
};
} // namespace mlCore::tensorInitializers

#endif