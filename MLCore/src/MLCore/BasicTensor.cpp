
#include <LoggingLib/LoggingLib.h>
#include <MLCore/BasicTensor.h>
#include <algorithm>
#include <exception>
#include <numeric>

namespace mlCore
{
template <typename valueType>
BasicTensor<valueType>::BasicTensor(const std::vector<uint64_t>& shape)
{
	if(std::any_of(shape.begin(), shape.end(), [](const auto dim) { return dim <= 0; }))
	{
		LOG_WARN("BasicTensor",
				 "Shape's members have to be greater than zero. Changing all zero dims to 1's.");

		std::for_each(shape.begin(), shape.end(), [](auto& dim) {
			if(dim == 0)
				dim = 1;
		});
	}

	length_ =
		std::accumulate(shape.begin(), shape.end(), 1, [](const auto current, const auto dim) {
			return current * dim;
		});

	shape_ = shape;

	data_ = new valueType[length_];
}

template <typename valueType>
BasicTensor<valueType>::BasicTensor(const std::vector<uint64_t>& shape, const valueType initialVal)
	: BasicTensor(shape)
{
	for(uint64_t i = 0; i < length_; i++)
	{
		data_[i] = initialVal;
	}
}

template <typename valueType>
BasicTensor<valueType>::BasicTensor(const BasicTensor& other)
{
	shape_ = other.shape_;
	length_ = other.length_;
	data_ = new valueType[length_];

	for(uint64_t i = 0; i < length_; i++)
	{
		data_[i] = other.data_[i];
	}
}

template <typename valueType>
BasicTensor<valueType>::BasicTensor(BasicTensor&& other)
{
	shape_ = std::move(other.shape_);

	length_ = other.length_;
	other.length_ = 0;

	data_ = other.data_;
	other.data_ = nullptr;
}

template <typename valueType>
BasicTensor<valueType>::~BasicTensor()
{
	delete data_;
}

template <typename valueType>
BasicTensor<valueType>& BasicTensor<valueType>::operator=(const BasicTensor& other)
{
	if(&other != this)
	{
		if(length_ != other.length_)
		{
			delete data_;
			length_ = other.length_;
			data_ = new valueType[length_];
		}

		shape_ = other.shape_;

		for(uint64_t i = 0; i < length_; i++)
		{
			data_[i] = other.data_[i];
		}
	}

	return *this;
}

template <typename valueType>
BasicTensor<valueType>& BasicTensor<valueType>::operator=(BasicTensor&& other)
{
	if(&other != this)
	{
		delete data_;
		data_ = other.data_;

		length_ = other.length_;
		other.length_ = 0;

		shape_ = std::move(other.shape_);
	}

	return *this;
}

template <typename valueType>
void BasicTensor<valueType>::reshape(const std::vector<uint64_t>& newShape)
{
	if(std::any_of(newShape.begin(), newShape.end(), [](const auto axis) { return axis <= 0; }))
	{
		throw std::invalid_argument("Shape's members have to be greater than zero.");
	}
	else if(const auto newLength =
				std::accumulate(newShape.begin(),
								newShape.end(),
								1,
								[](const auto current, const auto axis) { return current * axis; });
			newLength != length_)
	{
		throw std::out_of_range("Cannot reshape if new shape's total size (" +
								std::to_string(newLength) + ") does not match current (" +
								std::to_string(length_) + ")");
	}

	shape_ = newShape;
}

template <typename valueType>
void BasicTensor<valueType>::assign(std::initializer_list<std::pair<uint64_t, uint64_t>> indices,
									std::initializer_list<valueType> newData)
{ }
} // namespace mlCore
