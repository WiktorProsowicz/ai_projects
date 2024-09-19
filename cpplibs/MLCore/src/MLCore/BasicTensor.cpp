#include "MLCore/BasicTensor.h"

#include <climits>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <LoggingLib/LoggingLib.hpp>
#include <fmt/format.h>

#include "MLCore/TensorInitializers/ITensorInitializer.hpp"
#include "MLCore/TensorOperationsImpl.h"
#include "MLCore/Utilities.h"
#include "MLCore/UtilitiesImpl.h"

namespace mlCore
{
/**************************
 * Explicit instantiations
 **************************/
template class BasicTensor<double>;
template std::ostream& operator<<(std::ostream& ostream, const BasicTensor<double>& tensor);

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor()
	: _length(1)
	, _shape()
	, _data(new ValueType[1])
{}

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor(ValueType initVal)
	: BasicTensor()
{
	*_data = initVal;
}

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor(const std::vector<size_t>& shape)
	: _length()
	, _shape(shape)
	, _data()
{
	try
	{
		_checkShapeElementsPositive(_shape);
	}
	catch(const std::runtime_error&)
	{
		LOG_WARN("BasicTensor",
				 "Shape's members have to be greater than zero. Changing all zero dims to 1's.");

		std::for_each(_shape.begin(),
					  _shape.end(),
					  [](auto& dim)
					  {
						  if(dim == 0)
						  {
							  dim = 1;
						  }
					  });
	}

	try
	{
		_checkShapeFitsInBounds(_shape);
	}
	catch(const std::runtime_error&)
	{
		size_t nElements = 1;
		std::vector<size_t> newShape;

		for_each(shape.begin(),
				 shape.end(),
				 [&nElements, &newShape](const auto dim)
				 {
					 if(nElements <= (ULLONG_MAX / dim))
					 {
						 nElements *= dim;
						 newShape.push_back(dim);
					 }
				 });

		LOG_WARN("BasicTensor",
				 "Cumulative number of elements can't exceed the limit of ULL_MAX. Will preserve "
				 "shape: "
					 << detail::stringifyVector(newShape));

		_shape = std::move(newShape);
	}

	_length = std::accumulate(_shape.begin(),
							  _shape.end(),
							  size_t{1},
							  [](const auto current, const auto dim) { return current * dim; });

	_data = new ValueType[_length];
}

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor(const std::vector<size_t>& shape, const ValueType initVal)
	: BasicTensor(shape)
{
	for(size_t i = 0; i < _length; i++)
	{
		_data[i] = initVal;
	}
}

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor(const std::vector<size_t>& shape,
									const std::initializer_list<ValueType> initValues)
	: BasicTensor(shape)
{

	size_t pos = 0;
	for(auto valsIter = initValues.begin(); valsIter < initValues.end() && pos < _length; pos++, valsIter++)
	{
		_data[pos] = *valsIter;
	}
}

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor(const BasicTensor& other)
	: _length(other._length)
	, _shape(other._shape)
	, _data(new ValueType[_length])
{
	for(size_t pos = 0; pos < _length; pos++)
	{
		_data[pos] = other._data[pos];
	}
}

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor(BasicTensor&& other) noexcept
	: _length(other._length)
	, _shape(std::move(other._shape))
	, _data(other._data)
{
	other._length = 0;
	other._data = nullptr;
}

template <typename ValueType>
BasicTensor<ValueType>::~BasicTensor()
{
	delete[] _data;
}

template <typename ValueType>
BasicTensor<ValueType>& BasicTensor<ValueType>::operator=(const BasicTensor& other)
{
	if(&other != this)
	{
		if(_length != other._length)
		{
			delete[] _data;
			_length = other._length;
			_data = new ValueType[_length];
		}

		_shape = other._shape;

		for(size_t i = 0; i < _length; i++)
		{
			_data[i] = other._data[i];
		}
	}

	return *this;
}

template <typename ValueType>
BasicTensor<ValueType>& BasicTensor<ValueType>::operator=(BasicTensor&& other) noexcept
{
	if(&other != this)
	{
		delete[] _data;
		_data = other._data;
		other._data = nullptr;

		_length = other._length;
		other._length = 0;

		_shape = std::move(other._shape);
	}

	return *this;
}

template <typename ValueType>
void BasicTensor<ValueType>::reshape(const std::vector<size_t>& newShape)
{
	_checkShapeElementsPositive(newShape);
	_checkShapeFitsInBounds(newShape);
	_checkShapeCompatible(newShape);

	_shape = newShape;
}

template <typename ValueType>
void BasicTensor<ValueType>::_checkShapeElementsPositive(const std::vector<size_t>& shape)
{
	if(std::any_of(shape.begin(), shape.end(), [](const auto axis) { return axis <= 0; }))
	{
		throw std::runtime_error("Shape's members have to be greater than zero.");
	}
}

template <typename ValueType>
void BasicTensor<ValueType>::_checkShapeFitsInBounds(const std::vector<size_t>& shape)
{
	size_t nElements = 1;
	for_each(shape.begin(),
			 shape.end(),
			 [&nElements](const auto dim)
			 {
				 if(nElements <= (ULLONG_MAX / dim))
				 {
					 nElements *= dim;
				 }
				 else
				 {
					 throw std::runtime_error("Cumulative number of elements in shape exceeds ULLONG_MAX");
				 }
			 });
}

template <typename ValueType>
void BasicTensor<ValueType>::_checkShapeCompatible(const std::vector<size_t>& shape) const
{
	const auto newLength =
		std::accumulate(shape.begin(),
						shape.end(),
						size_t{0},
						[](const auto current, const auto axis) { return current * axis; });

	if(newLength != _length)
	{
		throw std::out_of_range(
			fmt::format("Cannot reshape if the new shape's total size ({}) does not match current the ({}).",
						newLength,
						_length));
	}
}

template <typename ValueType>
template <typename IndicesIter>
void BasicTensor<ValueType>::_checkIndicesList(IndicesIter beg, IndicesIter end) const
{
	if(beg == end)
	{
		throw std::out_of_range("Indices list must have minimum length of 1.");
	}

	if(static_cast<size_t>(std::distance(beg, end)) > _shape.size())
	{
		throw std::out_of_range("Indices list cannot be longer than tensor's shape.");
	}

	for(auto [indicesIt, shapeIt] = std::tuple{beg, size_t{0}}; indicesIt < end; ++indicesIt, ++shapeIt)
	{
		const auto& [lower, upper] = *indicesIt;

		if(upper <= lower)
		{
			throw std::out_of_range(
				fmt::format("Upper index is not greater than lower for shape '{}' at index {}.",
							detail::stringifyVector(_shape),
							shapeIt));
		}

		if(upper > _shape[shapeIt])
		{
			throw std::out_of_range(fmt::format(
				"Upper index cannot be greater than particular dimension size for shape '{}' at index {}.",
				detail::stringifyVector(_shape),
				shapeIt));
		}
	}
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensor<ValueType>::operator*(const BasicTensor& other) const
{
	auto ret = *this;
	detail::TensorOperationsImpl<double>::multiplyTensorsInPlace(ret, other);
	return ret;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensor<ValueType>::operator-(const BasicTensor& other) const
{
	auto ret = *this;
	detail::TensorOperationsImpl<double>::subtractTensorsInPlace(ret, other);
	return ret;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensor<ValueType>::operator+(const BasicTensor& other) const
{
	auto ret = *this;
	detail::TensorOperationsImpl<double>::addTensorsInPlace(ret, other);
	return ret;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensor<ValueType>::operator/(const BasicTensor& other) const
{
	auto ret = *this;
	detail::TensorOperationsImpl<double>::divideTensorsInPlace(ret, other);
	return ret;
}

template <typename ValueType>
BasicTensor<ValueType>& BasicTensor<ValueType>::operator+=(const BasicTensor& other)
{
	detail::TensorOperationsImpl<double>::addTensorsInPlace(*this, other);
	return *this;
}

template <typename ValueType>
BasicTensor<ValueType>& BasicTensor<ValueType>::operator-=(const BasicTensor& other)
{
	detail::TensorOperationsImpl<double>::subtractTensorsInPlace(*this, other);
	return *this;
}

template <typename ValueType>
BasicTensor<ValueType>& BasicTensor<ValueType>::operator*=(const BasicTensor& other)
{
	detail::TensorOperationsImpl<double>::multiplyTensorsInPlace(*this, other);
	return *this;
}

template <typename ValueType>
BasicTensor<ValueType>& BasicTensor<ValueType>::operator/=(const BasicTensor& other)
{
	detail::TensorOperationsImpl<double>::divideTensorsInPlace(*this, other);
	return *this;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensor<ValueType>::operator-() const
{
	BasicTensor<ValueType> ret(_shape);

	for(size_t i = 0; i < _length; i++)
	{
		ret._data[i] = -_data[i];
	}

	return ret;
}

template <typename ValueType>
void BasicTensor<ValueType>::assign(std::initializer_list<std::pair<size_t, size_t>> indices,
									std::initializer_list<ValueType> newData,
									const bool wrapData)
{
	_checkIndicesList(indices.begin(), indices.end());

	size_t itemsToAssign = 1;
	for(const auto& [lower, upper] : indices)
	{
		itemsToAssign *= (upper - lower);
	}

	if((itemsToAssign < newData.size()) && (!wrapData))
	{
		throw std::out_of_range("Too few values to assign to the tensor.");
	}

	std::vector<size_t> treePath;
	std::vector<size_t> maxPath;
	for(const auto& [lower, upper] : indices)
	{
		treePath.emplace_back(lower);
		maxPath.emplace_back(upper);
	}

	// for quicker iterating over data array when last dimensions are left with unspecified indices
	size_t wholeDimensionsOffset = 1;
	for(size_t i = _shape.size() - 1; i > treePath.size() - 1; i--)
	{
		wholeDimensionsOffset *= _shape[i];
	}

	auto computeFramePos =
		[&wholeDimensionsOffset](const std::vector<size_t>& treePath, const std::vector<size_t>& shape)
	{
		size_t offset = 0;
		size_t factor = wholeDimensionsOffset;
		for(size_t i = treePath.size() - 1; i < treePath.size(); i--)
		{
			offset += treePath[i] * factor;
			factor *= shape[i];
		}
		return offset;
	};

	auto dataIter = newData.begin();
	size_t elementsProcessed = 0;
	while(elementsProcessed < itemsToAssign)
	{
		auto dataPtr = _data + computeFramePos(treePath, _shape);

		for(size_t elemPos = 0; elemPos < wholeDimensionsOffset; elemPos++)
		{
			dataPtr[elemPos] = *dataIter;
			elementsProcessed++;
			dataIter++;
			if(dataIter == newData.end())
			{
				dataIter = newData.begin();
			}
		}

		for(size_t i = treePath.size() - 1; i < treePath.size(); i--)
		{
			treePath[i]++;
			if(treePath[i] < maxPath[i])
			{
				break;
			}

			treePath[i] = (indices.begin() + i)->first;
		}
	}
}

template <typename ValueType>
void BasicTensor<ValueType>::fill(const tensorInitializers::ITensorInitializer<ValueType>& initializer)
{
	size_t elementPos = 0;
	while(initializer.canYield() && (elementPos < _length))
	{
		_data[elementPos] = initializer.yield();
		elementPos++;
	}
	if(elementPos < _length)
	{
		throw std::out_of_range("Too few values to assign to the tensor");
	}
}

template <typename ValueType>
void BasicTensor<ValueType>::fill(std::initializer_list<ValueType> newData, const bool wrapData)
{
	fill(newData.begin(), newData.end(), wrapData);
}

template <typename ValueType>
BasicTensorSlice<ValueType>
BasicTensor<ValueType>::slice(const std::vector<std::pair<size_t, size_t>>& indices)
{
	_checkIndicesList(indices.begin(), indices.end());

	if(indices.size() == _shape.size())
	{
		return BasicTensorSlice<ValueType>(*this, indices);
	}

	std::vector<std::pair<size_t, size_t>> paddedIndices = indices;

	std::transform(std::next(_shape.cbegin(), static_cast<ptrdiff_t>(indices.size())),
				   _shape.cend(),
				   std::back_inserter(paddedIndices),
				   [](const auto& dim) { return std::pair<size_t, size_t>(0, dim); });

	return BasicTensorSlice<ValueType>(*this, paddedIndices);
}

template <typename TensorValueType>
std::ostream& operator<<(std::ostream& ostream, const BasicTensor<TensorValueType>& tensor)
{

	ostream << "<BasicTensor dtype=" << typeid(TensorValueType).name()
			<< " shape=" << detail::stringifyVector(tensor._shape) << ">";

	if(tensor.nDimensions() == 0)
	{
		ostream << "\n" << *tensor.begin();
		return ostream;
	}

	std::vector<std::string> stringifiedNumbers;
	std::transform(tensor.begin(),
				   tensor.end(),
				   std::back_inserter(stringifiedNumbers),
				   [](const auto& element) { return (std::ostringstream() << element).str(); });

	const auto longestElement = std::max_element(stringifiedNumbers.cbegin(),
												 stringifiedNumbers.cend(),
												 [](const auto& item1, const auto& item2)
												 { return (item1.length() < item2.length()); });

	const auto blockSize = static_cast<int>(longestElement->length());

	std::function<void(typename std::vector<size_t>::const_iterator,
					   std::vector<std::string>::const_iterator,
					   const std::string&,
					   size_t)>
		recursePrint;

	recursePrint = [&blockSize, &recursePrint, &ostream, &tensor](
					   std::vector<size_t>::const_iterator shapeIter,
					   std::vector<std::string>::const_iterator stringifiedDataIter,
					   const std::string& preamble,
					   size_t offset)
	{
		offset /= *shapeIter;
		if(shapeIter == std::prev(tensor._shape.end()))
		{
			ostream << "\n" << preamble << "[";

			size_t elementNr = 0;

			for(; elementNr < (*shapeIter) - 1; elementNr++)
			{
				ostream << std::setw(blockSize)
						<< *std::next(stringifiedDataIter, static_cast<ptrdiff_t>(elementNr)) << ", ";
			}

			ostream << std::setw(blockSize)
					<< *std::next(stringifiedDataIter, static_cast<ptrdiff_t>(elementNr));

			ostream << "]";
		}
		else
		{
			ostream << "\n" << preamble << "[";

			size_t printNr = 0;
			for(; printNr < (*shapeIter); printNr++)
			{
				recursePrint(shapeIter + 1,
							 std::next(stringifiedDataIter, static_cast<ptrdiff_t>(printNr * offset)),
							 preamble + " ",
							 offset);
			}

			ostream << "\n" << preamble << "]";
		}
	};

	// for traversing data
	const auto offset = std::accumulate(tensor._shape.begin(),
										tensor._shape.end(),
										size_t{1},
										[](const size_t& curr, const size_t& dim) { return curr * dim; });

	recursePrint(tensor._shape.cbegin(), stringifiedNumbers.begin(), "", offset);

	return ostream;
}

} // namespace mlCore
