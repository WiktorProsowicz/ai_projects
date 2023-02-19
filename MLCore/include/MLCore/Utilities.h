#ifndef MLCORE_UTILITIES_H
#define MLCORE_UTILITIES_H

#include <iterator>
#include <limits>
#include <sstream>
#include <vector>

namespace mlCore
{
template <typename T>
std::string stringifyVector(const std::vector<T>& vect,
							const char* const openSign = "(",
							const char* const closeSign = ")")
{
	std::ostringstream ss;
	ss << openSign;
	std::copy(vect.cbegin(), vect.cend(), std::ostream_iterator<T>(ss, ", "));
	ss << closeSign;
	return ss.str();
}

template <typename T>
struct ITensorInitializer
{
	virtual const T yield() = 0;
	virtual bool canYield()
	{
		return true;
	}
	virtual ~ITensorInitializer() = default;
};

template <class T>
struct RangeTensorInitializer : public ITensorInitializer<T>
{
	RangeTensorInitializer(T _firstValue, T _maxValue = std::numeric_limits<T>::max(), T _step = 1)
		: currentValue(_firstValue)
		, step(_step)
		, maxValue(_maxValue)
	{ }
	virtual const T yield() override
	{
		if(!canYield())
			throw std::out_of_range("Cannot obtain value from RangeTensorYielder.");

		const T out = currentValue;
		currentValue += step;
		return out;
	}
	virtual bool canYield() override
	{
		return currentValue < maxValue;
	}

private:
	T currentValue;
	T maxValue;
	T step;
};

} // namespace mlCore
#endif