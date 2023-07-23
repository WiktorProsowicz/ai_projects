#include <MLCore/TensorInitializers/RangeTensorInitializer.h>

#include <stdexcept>

namespace mlCore::tensorInitializers
{
template class RangeTensorInitializer<double>;

template <typename ValueType>
RangeTensorInitializer<ValueType>::RangeTensorInitializer(ValueType firstValue,
														  ValueType step,
														  ValueType maxValue)
	: currentValue_(firstValue)
	, maxValue_(maxValue)
	, step_(step)

{ }

template <typename ValueType>
ValueType RangeTensorInitializer<ValueType>::yield() const
{
	if(!canYield())
	{
		throw std::out_of_range("Cannot obtain value from RangeTensorYielder.");
	}

	ValueType out = currentValue_;
	currentValue_ += step_;
	return out;
}

template <typename ValueType>
bool RangeTensorInitializer<ValueType>::canYield() const
{
	return currentValue_ <= maxValue_;
}
} // namespace mlCore::tensorInitializers