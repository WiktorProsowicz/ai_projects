
#include "MLCore/BasicTensor.h"
#include <MLCore/TensorOperationsImpl.h>

namespace mlCore
{
/**************************
 * Explicit instantiations
 **************************/
template class BasicTensor<double>;
template std::ostream& operator<<(std::ostream& out, const BasicTensor<double>& tensor);

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor()
	: length_(1)
	, shape_()
	, data_()
{
	data_ = new ValueType[1];
}

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor(const std::vector<size_t>& shape)
	: length_()
	, shape_(shape)
	, data_()
{
	try
	{
		_checkShapeElementsPositive(shape_);
	}
	catch(const std::runtime_error&)
	{
		LOG_WARN("BasicTensor",
				 "Shape's members have to be greater than zero. Changing all zero dims to 1's.");

		std::for_each(shape_.begin(), shape_.end(), [](auto& dim) {
			if(dim == 0)
			{
				dim = 1;
			}
		});
	}

	try
	{
		_checkShapeFitsInBounds(shape_);
	}
	catch(const std::runtime_error&)
	{
		size_t nElements = 1;
		std::vector<size_t> newShape;

		for_each(shape.begin(), shape.end(), [&nElements, &newShape](const auto dim) {
			if(nElements <= (ULLONG_MAX / dim))
			{
				nElements *= dim;
				newShape.push_back(dim);
			}
		});

		LOG_WARN("BasicTensor",
				 "Cumulative number of elements can't exceed the limit of ULL_MAX. Will preserve "
				 "shape: "
					 << stringifyVector(newShape));

		shape_ = std::move(newShape);
	}

	length_ = std::accumulate(shape_.begin(),
							  shape_.end(),
							  size_t(1),
							  [](const auto current, const auto dim) { return current * dim; });

	data_ = new ValueType[length_];
}

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor(const std::vector<size_t>& shape, const ValueType initVal)
	: BasicTensor(shape)
{
	for(size_t i = 0; i < length_; i++)
	{
		data_[i] = initVal;
	}
}

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor(const std::vector<size_t>& shape,
									const std::initializer_list<ValueType> initValues)
	: BasicTensor(shape)
{

	size_t pos = 0;
	for(auto valsIter = initValues.begin(); valsIter < initValues.end() && pos < length_;
		pos++, valsIter++)
	{
		data_[pos] = *valsIter;
	}
}

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor(const BasicTensor& other)
	: length_(other.length_)
	, shape_(other.shape_)
	, data_(new ValueType[length_])
{
	for(size_t pos = 0; pos < length_; pos++)
	{
		data_[pos] = other.data_[pos];
	}
}

template <typename ValueType>
BasicTensor<ValueType>::BasicTensor(BasicTensor&& other)
	: length_(other.length_)
	, shape_(std::move(other.shape_))
	, data_(other.data_)
{
	other.length_ = 0;
	other.data_ = nullptr;
}

template <typename ValueType>
BasicTensor<ValueType>::~BasicTensor()
{
	delete[] data_;
}

template <typename ValueType>
BasicTensor<ValueType>& BasicTensor<ValueType>::operator=(const BasicTensor& other)
{
	if(&other != this)
	{
		if(length_ != other.length_)
		{
			delete[] data_;
			length_ = other.length_;
			data_ = new ValueType[length_];
		}

		shape_ = other.shape_;

		for(size_t i = 0; i < length_; i++)
		{
			data_[i] = other.data_[i];
		}
	}

	return *this;
}

template <typename ValueType>
BasicTensor<ValueType>& BasicTensor<ValueType>::operator=(BasicTensor&& other)
{
	if(&other != this)
	{
		delete[] data_;
		data_ = other.data_;
		other.data_ = nullptr;

		length_ = other.length_;
		other.length_ = 0;

		shape_ = std::move(other.shape_);
	}

	return *this;
}

template <typename ValueType>
void BasicTensor<ValueType>::reshape(const std::vector<size_t>& newShape)
{
	_checkShapeElementsPositive(newShape);
	_checkShapeFitsInBounds(newShape);
	_checkShapeCompatible(newShape);

	shape_ = newShape;
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
	for_each(shape.begin(), shape.end(), [&nElements](const auto dim) {
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
	const auto newLength = std::accumulate(
		shape.begin(), shape.end(), size_t(0), [](const auto current, const auto axis) {
			return current * axis;
		});

	if(newLength != length_)
	{
		throw std::out_of_range("Cannot reshape if new shape's total size (" +
								std::to_string(newLength) + ") does not match current (" +
								std::to_string(length_) + ")");
	}
}

template <typename ValueType>
void BasicTensor<ValueType>::_checkIndicesList(
	const std::initializer_list<std::pair<size_t, size_t>>::const_iterator _beg,
	const std::initializer_list<std::pair<size_t, size_t>>::const_iterator _end) const
{
	if(_beg == _end)
	{
		throw std::out_of_range("Indices list must have minimum length of 1.");
	}

	if(static_cast<size_t>(std::distance(_beg, _end)) > shape_.size())
	{
		throw std::out_of_range("Indices list cannot be longer than tensor's shape.");
	}

	for(auto [indicesIt, shapeIt] = std::tuple{_beg, size_t(0)}; indicesIt < _end;
		++indicesIt, ++shapeIt)
	{
		const auto& [lower, upper] = *indicesIt;

		if(upper <= lower)
		{
			throw std::out_of_range(
				std::string("Upper index is not greater than lower for shape '") +
				stringifyVector(shape_) + "' at index " + std::to_string(shapeIt) + ".");
		}

		if(upper > shape_[shapeIt])
		{
			throw std::out_of_range(
				std::string(
					"Upper index cannot be greater than particular dimension size for shape '") +
				stringifyVector(shape_) + "' at index " + std::to_string(shapeIt) + ".");
		}
	}
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensor<ValueType>::operator*(const BasicTensor& other) const
{
	auto ret = *this;
	TensorOperationsImpl<double>::multiplyTensorsInPlace(ret, other);
	return ret;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensor<ValueType>::operator-(const BasicTensor& other) const
{
	auto ret = *this;
	TensorOperationsImpl<double>::subtractTensorsInPlace(ret, other);
	return ret;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensor<ValueType>::operator+(const BasicTensor& other) const
{
	auto ret = *this;
	TensorOperationsImpl<double>::addTensorsInPlace(ret, other);
	return ret;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensor<ValueType>::operator/(const BasicTensor& other) const
{
	auto ret = *this;
	TensorOperationsImpl<double>::divideTensorsInPlace(ret, other);
	return ret;
}

template <typename ValueType>
BasicTensor<ValueType>& BasicTensor<ValueType>::operator+=(const BasicTensor& other)
{
	TensorOperationsImpl<double>::addTensorsInPlace(*this, other);
	return *this;
}

template <typename ValueType>
BasicTensor<ValueType>& BasicTensor<ValueType>::operator-=(const BasicTensor& other)
{
	TensorOperationsImpl<double>::subtractTensorsInPlace(*this, other);
	return *this;
}

template <typename ValueType>
BasicTensor<ValueType>& BasicTensor<ValueType>::operator*=(const BasicTensor& other)
{
	TensorOperationsImpl<double>::multiplyTensorsInPlace(*this, other);
	return *this;
}

template <typename ValueType>
BasicTensor<ValueType>& BasicTensor<ValueType>::operator/=(const BasicTensor& other)
{
	TensorOperationsImpl<double>::divideTensorsInPlace(*this, other);
	return *this;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensor<ValueType>::operator-() const
{
	BasicTensor<ValueType> ret(shape_);

	for(size_t i = 0; i < length_; i++)
	{
		ret.data_[i] = -data_[i];
	}

	return ret;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensor<ValueType>::matmul(const BasicTensor& other) const
{
	// for clean error throwing with additional info about shapes
	auto throwInformative = [this, &other](const std::string& message) {
		throw std::invalid_argument(
			"Can't perform matrix multiplication for shapes: " + stringifyVector(shape_) + ", " +
			stringifyVector(other.shape_) + " - " + message);
	};

	// checking if first tensor can be padded to nDims >= 2
	if(other.shape_.size() < 2)
	{
		throwInformative("cannot obtain last but one dimension of the second tensor.");
	}

	// for padding tensors shapes
	const size_t biggerSize = std::max(shape_.size(), other.shape_.size());

	// padded shapes for easier operating
	std::vector<size_t> paddedShapeFirst(biggerSize, 1);
	std::vector<size_t> paddedShapeSecond(biggerSize, 1);

	std::copy(
		shape_.cbegin(),
		shape_.cend(),
		std::next(paddedShapeFirst.begin(), static_cast<ptrdiff_t>(biggerSize - shape_.size())));

	std::copy(other.shape_.cbegin(),
			  other.shape_.cend(),
			  std::next(paddedShapeSecond.begin(),
						static_cast<ptrdiff_t>(biggerSize - other.shape_.size())));

	// checking matmul conditions
	if(paddedShapeFirst[biggerSize - 1] != paddedShapeSecond[biggerSize - 2])
	{
		throwInformative("last two dimensions are incompatible.");
	}

	for(size_t i = 0; i < biggerSize - 2; i++)
	{
		if((paddedShapeFirst[i] != paddedShapeSecond[i]) && (paddedShapeFirst[i] != 1) &&
		   (paddedShapeSecond[i] != 1))
		{
			throwInformative("shapes are incompatible.");
		}
	}

	std::vector<size_t> retShape(biggerSize);
	retShape[biggerSize - 2] = paddedShapeFirst[biggerSize - 2];
	retShape[biggerSize - 1] = paddedShapeSecond[biggerSize - 1];

	for(size_t i = 0; i < biggerSize - 2; i++)
	{
		retShape[i] = paddedShapeFirst[i] == 1 ? paddedShapeSecond[i] : paddedShapeFirst[i];
	}

	BasicTensor<ValueType> resultTensor(retShape, 0);

	// tells the position of a single computed matrix relative to the array of values
	auto computeFramePos = [](const std::vector<size_t>& treePath,
							  const std::vector<size_t>& shape) -> size_t {
		size_t offset = 0;
		size_t factor = shape[shape.size() - 1] * shape[shape.size() - 2];
		for(size_t i = shape.size() - 3; i < shape.size() - 2; i--)
		{
			offset += treePath[i] * factor;
			factor *= shape[i];
		}
		return offset;
	};

	size_t resElementPos = 0;
	const size_t adjacentDimension = paddedShapeFirst[biggerSize - 1];
	std::vector<size_t> firstTreePath(biggerSize - 2, 0);
	std::vector<size_t> secondTreePath(biggerSize - 2, 0);

	while(resElementPos < resultTensor.length_)
	{
		// proper data pointers with offsets for obtaining frame elements
		const ValueType* firstDataPtr = data_ + computeFramePos(firstTreePath, shape_);
		const ValueType* secondDataPtr =
			other.data_ + computeFramePos(secondTreePath, other.shape_);

		// rows and cols of result frame
		for(size_t rowIter = 0; rowIter < retShape[biggerSize - 2]; rowIter++)
		{
			for(size_t colIter = 0; colIter < retShape[biggerSize - 1]; colIter++)
			{
				// position of a certain multiplication in sum of multiplications
				for(size_t mulIter = 0; mulIter < adjacentDimension; mulIter++)
				{
					resultTensor.data_[resElementPos] +=
						firstDataPtr[rowIter * adjacentDimension + mulIter] *
						secondDataPtr[mulIter * paddedShapeSecond[biggerSize - 1] + colIter];
				}
				resElementPos++;
			}
		}

		// tells which dimension of the tree path should be incremented
		for(size_t i = biggerSize - 3; i < biggerSize - 2; i--)
		{

			if(paddedShapeFirst[i] > 1)
			{
				firstTreePath[i]++;
			}

			if(paddedShapeSecond[i] > 1)
			{
				secondTreePath[i]++;
			}

			if((firstTreePath[i] < paddedShapeFirst[i]) &&
			   (secondTreePath[i] < paddedShapeSecond[i]))
			{
				break;
			}

			firstTreePath[i] = 0;
			secondTreePath[i] = 0;
		}
	}

	return resultTensor;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensor<ValueType>::transposed() const
{

	std::vector<size_t> retShape;
	for(size_t i = 0; i < shape_.size() - 2; i++)
	{
		retShape.push_back(shape_[i]);
	}

	// size of a single 2-dimensional part that takes part in single matrix multiplication
	const size_t frameShapeFirst = *(++shape_.rbegin());
	const size_t frameShapeSecond = *shape_.rbegin();
	const size_t frameLength = frameShapeFirst * frameShapeSecond;

	retShape.push_back(frameShapeSecond);
	retShape.push_back(frameShapeFirst);

	BasicTensor<ValueType> ret(retShape);

	for(size_t frameOffset = 0; frameOffset < length_; frameOffset += frameLength)
	{
		for(size_t posInFrame = 0; posInFrame < frameLength; posInFrame++)
		{
			ret.data_[frameOffset + posInFrame] =
				data_[frameOffset + (posInFrame % frameShapeFirst) * frameShapeSecond +
					  (posInFrame / frameShapeFirst)];
		}
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
	for(size_t i = shape_.size() - 1; i > treePath.size() - 1; i--)
	{
		wholeDimensionsOffset *= shape_[i];
	}

	auto computeFramePos = [&wholeDimensionsOffset](const std::vector<size_t>& treePath,
													const std::vector<size_t>& shape) {
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
		auto dataPtr = data_ + computeFramePos(treePath, shape_);

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
void BasicTensor<ValueType>::fill(
	const tensorInitializers::ITensorInitializer<ValueType>& initializer)
{
	size_t elementPos = 0;
	while(initializer.canYield() && (elementPos < length_))
	{
		data_[elementPos] = initializer.yield();
		elementPos++;
	}
	if(elementPos < length_)
	{
		throw std::out_of_range("Too few values to assign to the tensor.");
	}
}

template <typename ValueType>
void BasicTensor<ValueType>::fill(std::initializer_list<ValueType> newData, const bool wrapData)
{
	fill(newData.begin(), newData.end(), wrapData);
}

template <typename TensorValueType>
std::ostream& operator<<(std::ostream& out, const BasicTensor<TensorValueType>& tensor)
{

	out << "<BasicTensor dtype=" << typeid(TensorValueType).name()
		<< " shape=" << stringifyVector(tensor.shape_) << ">";

	if(tensor.nDimensions() == 0)
	{
		out << "[\n " << *tensor.begin() << "\n]";
		return out;
	}

	std::vector<std::string> stringifiedNumbers;
	std::transform(tensor.begin(),
				   tensor.end(),
				   std::back_inserter(stringifiedNumbers),
				   [](const auto& element) { return (std::ostringstream() << element).str(); });

	const auto longestElement = std::max_element(
		stringifiedNumbers.cbegin(),
		stringifiedNumbers.cend(),
		[](const auto& item1, const auto& item2) { return (item1.length() < item2.length()); });

	const auto blockSize = static_cast<int>(longestElement->length());

	std::function<void(typename std::vector<size_t>::const_iterator,
					   std::vector<std::string>::const_iterator,
					   const std::string&,
					   size_t)>
		recursePrint;

	recursePrint = [&blockSize, &recursePrint, &out, &tensor](
					   std::vector<size_t>::const_iterator shapeIter,
					   std::vector<std::string>::const_iterator stringifiedDataIter,
					   const std::string& preamble,
					   size_t offset) {
		offset /= *shapeIter;
		if(shapeIter == std::prev(tensor.shape_.end()))
		{
			out << "\n" << preamble << "[";

			size_t elementNr;

			for(elementNr = 0; elementNr < (*shapeIter) - 1; elementNr++)
			{
				out << std::setw(blockSize)
					<< *std::next(stringifiedDataIter, static_cast<ptrdiff_t>(elementNr)) << ", ";
			}

			out << std::setw(blockSize)
				<< *std::next(stringifiedDataIter, static_cast<ptrdiff_t>(elementNr));

			out << "]";
		}
		else
		{
			out << "\n" << preamble << "[";

			size_t printNr;
			for(printNr = 0; printNr < (*shapeIter); printNr++)
			{
				recursePrint(
					shapeIter + 1,
					std::next(stringifiedDataIter, static_cast<ptrdiff_t>(printNr * offset)),
					preamble + " ",
					offset);
			}

			out << "\n" << preamble << "]";
		}
	};

	// for traversing data
	const auto offset =
		std::accumulate(tensor.shape_.begin(),
						tensor.shape_.end(),
						size_t(1),
						[](const size_t& curr, const size_t& dim) { return curr * dim; });

	recursePrint(tensor.shape_.cbegin(), stringifiedNumbers.begin(), "", offset);

	return out;
}

} // namespace mlCore
