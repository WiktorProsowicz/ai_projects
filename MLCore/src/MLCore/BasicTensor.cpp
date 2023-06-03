
#include "MLCore/BasicTensor.h"

namespace mlCore
{
// explicit instantiation
template class BasicTensor<double>;

template <typename valueType>
BasicTensor<valueType>::BasicTensor(const std::vector<size_t>& shape)
{
	// if shape checking chain succeeds, this value will stay unchanged
	shape_ = shape;

	try
	{
		// shape checking chain
		checkShape_(shape);
	}
	catch(const std::invalid_argument&)
	{
		LOG_WARN("BasicTensor",
				 "Shape's members have to be greater than zero. Changing all zero dims to 1's.");

		std::for_each(shape_.begin(), shape_.end(), [](auto& dim) {
			if(dim == 0)
				dim = 1;
		});
	}
	// last link in chain, will always be thrown
	catch(const std::out_of_range&)
	{ }
	catch(const std::length_error&)
	{
		size_t nElements = 1;
		std::string newShapeRepr = "(";

		for_each(shape.begin(), shape.end(), [&nElements, &newShapeRepr](const auto dim) {
			if(nElements <= (ULLONG_MAX / dim))
			{
				newShapeRepr += std::to_string(dim) + ",";
				nElements *= dim;
			}
		});

		newShapeRepr += ")";
		LOG_WARN("BasicTensor",
				 "Cummulative number of elements can't exceed the limit of ULL_MAX. Will preserve "
				 "shape: " +
					 newShapeRepr);
	}

	length_ =
		std::accumulate(shape.begin(), shape.end(), 1, [](const auto current, const auto dim) {
			return current * dim;
		});

	data_ = new valueType[length_];
}

template <typename valueType>
BasicTensor<valueType>::BasicTensor(const std::vector<size_t>& shape, const valueType initialVal)
	: BasicTensor(shape)
{
	for(size_t i = 0; i < length_; i++)
	{
		data_[i] = initialVal;
	}
}

template <typename valueType>
BasicTensor<valueType>::BasicTensor(const std::vector<size_t>& shape,
									const std::initializer_list<valueType> initValues)
	: BasicTensor(shape)
{

	size_t i = 0;
	for(auto valsIter = initValues.begin(); valsIter < initValues.end() && i < length_;
		i++, valsIter++)
	{
		data_[i] = *valsIter;
	}
}

template <typename valueType>
BasicTensor<valueType>::BasicTensor(const BasicTensor& other)
{
	shape_ = other.shape_;
	length_ = other.length_;
	data_ = new valueType[length_];

	for(size_t i = 0; i < length_; i++)
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
	if(data_)
		delete[] data_;
}

template <typename valueType>
BasicTensor<valueType>& BasicTensor<valueType>::operator=(const BasicTensor& other)
{
	if(&other != this)
	{
		if(length_ != other.length_)
		{
			delete[] data_;
			length_ = other.length_;
			data_ = new valueType[length_];
		}

		shape_ = other.shape_;

		for(size_t i = 0; i < length_; i++)
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
		delete[] data_;
		data_ = other.data_;
		other.data_ = nullptr;

		length_ = other.length_;
		other.length_ = 0;

		shape_ = std::move(other.shape_);
	}

	return *this;
}

template <typename valueType>
void BasicTensor<valueType>::reshape(const std::vector<size_t>& newShape)
{
	checkShape_(newShape);
	shape_ = newShape;
}

template <typename valueType>
void BasicTensor<valueType>::checkShape_(const std::vector<size_t>& shape) const
{
	if(std::any_of(shape.begin(), shape.end(), [](const auto axis) { return axis <= 0; }))
	{
		throw std::invalid_argument("Shape's members have to be greater than zero.");
	}

	// counts all elements given by the shape to check validity
	size_t nElements = 1;
	for_each(shape.begin(), shape.end(), [&nElements](const auto dim) {
		if(nElements <= (ULLONG_MAX / dim))
		{
			nElements *= dim;
		}
		else
		{
			throw std::length_error("Cummulative number of elements in shape exceeds ULLONG_MAX");
		}
	});

	if(const auto newLength =
		   std::accumulate(shape.begin(),
						   shape.end(),
						   size_t(0),
						   [](const auto current, const auto axis) { return current * axis; });
	   newLength != length_)
	{
		throw std::out_of_range("Cannot reshape if new shape's total size (" +
								std::to_string(newLength) + ") does not match current (" +
								std::to_string(length_) + ")");
	}
}

/**
 * @brief traverses list of indices and checks ranges correctness. Correct indices specify tensor slice that can be modified via value assignment.
 * Throws std::out_of_range if upper[i] > shape[i] or 0 > indices.size() > shape_.size()
 * 
 * @tparam valueType dtype of tensor
 * @param indices list of pairs of min-max indices from axis zero i.e for tensor([[1, 2], [3, 4]]) -> list{{0, 1}} -> [1, 2]
 */
template <typename valueType>
void BasicTensor<valueType>::checkIndicesList_(
	const std::initializer_list<std::pair<size_t, size_t>>::const_iterator _beg,
	const std::initializer_list<std::pair<size_t, size_t>>::const_iterator _end) const
{
	if(_beg == _end)
		throw std::out_of_range("Indices list must have minimum length of 1.");

	if(std::distance(_beg, _end) > shape_.size())
		throw std::out_of_range("Indices list cannot be longer than tensor's shape.");

	for(auto [it, i] = std::tuple{_beg, 0}; it < _end; ++it, ++i)
	{
		const auto& [lower, upper] = *it;
		if(upper <= lower)
			throw std::out_of_range(
				std::string("Upper index is not greater than lower for shape '") +
				stringifyVector(shape_) + "' at index " + std::to_string(i) + ".");

		if(upper > shape_[i])
			throw std::out_of_range(
				std::string(
					"Upper index cannot be greater than particular dimension size for shape '") +
				stringifyVector(shape_) + "' at index " + std::to_string(i) + ".");
	}
}

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::operator*(const BasicTensor& other) const
{
	if(shape_ == other.shape_)
	{
		BasicTensor<valueType> ret(shape_);
		for(size_t i = 0; i < length_; i++)
			ret.data_[i] = data_[i] * other.data_[i];

		return ret;
	}
	return performOperation_(other, mulOperator_);
}

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::operator-(const BasicTensor& other) const
{
	if(shape_ == other.shape_)
	{
		BasicTensor<valueType> ret(shape_);
		for(size_t i = 0; i < length_; i++)
			ret.data_[i] = data_[i] - other.data_[i];

		return ret;
	}
	return performOperation_(other, minusOperator_);
}

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::operator+(const BasicTensor& other) const
{
	if(shape_ == other.shape_)
	{
		BasicTensor<valueType> ret(shape_);
		for(size_t i = 0; i < length_; i++)
			ret.data_[i] = data_[i] + other.data_[i];

		return ret;
	}
	return performOperation_(other, plusOperator_);
}

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::operator/(const BasicTensor& other) const
{
	if(shape_ == other.shape_)
	{
		BasicTensor<valueType> ret(shape_);
		for(size_t i = 0; i < length_; i++)
			ret.data_[i] = data_[i] / other.data_[i];

		return ret;
	}
	return performOperation_(other, divOperator_);
}

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::operator-() const
{
	BasicTensor<valueType> ret(shape_);

	for(size_t i = 0; i < length_; i++)
		ret.data_[i] = -data_[i];

	return ret;
}

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::matmul(const BasicTensor& other) const
{
	// for clean error throwing with additional info about shapes
	auto throwInformative = [this, &other](const std::string& m) {
		throw std::invalid_argument(
			"Can't perform matrix multipliaction for shapes: " + stringifyVector(shape_) + ", " +
			stringifyVector(other.shape_) + " - " + m);
	};

	// checking if first tensor can be padded to nDims >= 2
	if(other.shape_.size() < 2)
		throwInformative("cannot obtain last but one dimension of the second tensor.");

	// for padding tensors shapes
	const size_t biggerSize = std::max(shape_.size(), other.shape_.size());

	// padded shapes for easier operating
	std::vector<size_t> paddedShapeFirst(biggerSize, 1), paddedShapeSecond(biggerSize, 1);
	std::copy(
		shape_.cbegin(), shape_.cend(), paddedShapeFirst.begin() + biggerSize - shape_.size());
	std::copy(other.shape_.cbegin(),
			  other.shape_.cend(),
			  paddedShapeSecond.begin() + biggerSize - other.shape_.size());

	// checking matmul conditions
	if(paddedShapeFirst[biggerSize - 1] != paddedShapeSecond[biggerSize - 2])
	{
		throwInformative("last two dimensions are incompatibile.");
	}

	for(size_t i = 0; i < biggerSize - 2; i++)
	{
		if((paddedShapeFirst[i] != paddedShapeSecond[i]) && (paddedShapeFirst[i] != 1) &&
		   (paddedShapeSecond[i] != 1))
			throwInformative("shapes are incompatibile.");
	}

	std::vector<size_t> retShape(biggerSize);
	retShape[biggerSize - 2] = paddedShapeFirst[biggerSize - 2];
	retShape[biggerSize - 1] = paddedShapeSecond[biggerSize - 1];

	for(size_t i = 0; i < biggerSize - 2; i++)
	{
		retShape[i] = paddedShapeFirst[i] == 1 ? paddedShapeSecond[i] : paddedShapeFirst[i];
	}

	BasicTensor<valueType> resultTensor(retShape, 0);

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
	std::vector<size_t> firstTreePath(biggerSize - 2, 0), secondTreePath(biggerSize - 2, 0);

	while(resElementPos < resultTensor.length_)
	{
		// proper data pointers with offsets for obtaining frame elements
		const valueType* firstDataPtr = data_ + computeFramePos(firstTreePath, shape_);
		const valueType* secondDataPtr =
			other.data_ + computeFramePos(secondTreePath, other.shape_);

		// rows and cols of result frame
		for(size_t rowIter = 0; rowIter < retShape[biggerSize - 2]; rowIter++)
		{
			for(size_t colIter = 0; colIter < retShape[biggerSize - 1]; colIter++)
			{
				// positon of a certain multiplication in sum of multiplications
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
				firstTreePath[i]++;

			if(paddedShapeSecond[i] > 1)
				secondTreePath[i]++;

			if((firstTreePath[i] < paddedShapeFirst[i]) &&
			   (secondTreePath[i] < paddedShapeSecond[i]))
				break;

			firstTreePath[i] = 0;
			secondTreePath[i] = 0;
		}
	}

	return resultTensor;
}

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::transposed() const
{

	std::vector<size_t> retShape;
	for(size_t i = 0; i < shape_.size() - 2; i++)
		retShape.push_back(shape_[i]);

	// size of a single 2-dimensional part that takes part in single matrix multiplication
	const size_t frameShapeFirst = *(++shape_.rbegin()), frameShapeSecond = *shape_.rbegin();
	const size_t frameLength = frameShapeFirst * frameShapeSecond;

	retShape.push_back(frameShapeSecond);
	retShape.push_back(frameShapeFirst);

	BasicTensor<valueType> ret(retShape);

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

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::performOperation_(
	const BasicTensor<valueType>& other,
	const std::function<valueType(const valueType, const valueType)>& op_) const
{
	// checking if the rules of broadcasting are not breached
	auto selfShapeIter = shape_.rbegin();
	auto otherShapeIter = other.shape_.rbegin();
	for(; (selfShapeIter > shape_.rend()) && (otherShapeIter > other.shape_.rend());
		selfShapeIter--, otherShapeIter--)
	{
		if((*selfShapeIter != 1) && (*otherShapeIter != 1) && (*selfShapeIter != *otherShapeIter))
			throw std::invalid_argument(
				"Can't perform broadcasting operation on tensors with invalid shapes: " +
				stringifyVector(shape_) + " " + stringifyVector(other.shape_));
	}

	const auto biggerSize = std::max(shape_.size(), other.shape_.size());

	// shapes after padding with ones
	std::vector<size_t> paddedLeftShape(biggerSize, 1), paddedRightShape(biggerSize, 1);

	std::copy(shape_.cbegin(), shape_.cend(), paddedLeftShape.begin() + biggerSize - shape_.size());
	std::copy(other.shape_.cbegin(),
			  other.shape_.cend(),
			  paddedRightShape.begin() + biggerSize - other.shape_.size());

	std::vector<size_t> retShape(biggerSize);
	for(size_t i = 0; i < biggerSize; i++)
	{
		// filling new shape according to dimension that can be stretched
		retShape[i] = paddedLeftShape[i] == 1 ? paddedRightShape[i] : paddedLeftShape[i];
	}

	BasicTensor<valueType> ret(retShape, 0);

	// handles stretching smaller tensor into returned one's payload
	// clang-format off

	// determines the being-stretched tensor's current copied element
	std::vector<size_t> factorTreePath(biggerSize, 0);
	// determines the place for added element in returned tensor
	std::vector<size_t> destTreePath(biggerSize, 0);

	auto computeElementPosition = [](const std::vector<size_t>& path, const std::vector<size_t>& shape) -> size_t
	{
		// specifies the offset added to total position while traversing respecitve dimensions
		size_t offset = 1;
		size_t position = 0;
		for(size_t i = path.size() - 1; i < path.size(); i--)
		{
			position += offset * path[i];
			offset *= shape[i];
		}

		return position;
	};

	// updates tree paths untill all elements are processed and constantly assigns proper elements to proper places
	std::function<void(valueType* const, const valueType* const, 
					   const std::vector<size_t>&, 
					   const std::function<valueType(const valueType, const valueType)>&)> appendTensor;

	appendTensor = [&factorTreePath, &destTreePath, &retShape, &computeElementPosition, &ret](valueType* const destDataPtr,
															   const valueType* const factorDataPtr,
															   const std::vector<size_t>& factorShape,
															   const std::function<valueType(const valueType, const valueType)>& op) mutable
	{
		size_t elementsProcessed = 0;
		size_t destTensorLength = std::accumulate(retShape.cbegin(), retShape.cend(), 1, 
													[](const auto curr, const auto dim){ return curr * dim; });

		while(elementsProcessed < destTensorLength)
		{
			const auto destElemPos = computeElementPosition(destTreePath, retShape);
			const auto factorElemPos = computeElementPosition(factorTreePath, factorShape);

			destDataPtr[destElemPos] = op(destDataPtr[destElemPos], factorDataPtr[factorElemPos]);

			elementsProcessed++;

			for(size_t i = retShape.size() - 1; i < retShape.size(); i--) {

				
				destTreePath[i]++;
				if(factorShape[i] > 1)
					factorTreePath[i]++;

				if(destTreePath[i] < retShape[i])
					break;

				destTreePath[i] = 0;
				factorTreePath[i] = 0;
			}
		}
	};
	// clang-format on

	appendTensor(ret.data_, this->data_, paddedLeftShape, plusOperator_);

	for(size_t i = 0; i < biggerSize; i++)
	{
		destTreePath[i] = 0;
		factorTreePath[i] = 0;
	}

	appendTensor(ret.data_, other.data_, paddedRightShape, op_);

	return ret;
}

template <typename valueType>
void BasicTensor<valueType>::assign(std::initializer_list<std::pair<size_t, size_t>> indices,
									std::initializer_list<valueType> newData,
									const bool wrapData)
{
	checkIndicesList_(indices.begin(), indices.end());

	size_t itemsToAssign = 1;
	for(const auto& [lower, upper] : indices)
	{
		itemsToAssign *= (upper - lower);
	}

	if((itemsToAssign < newData.size()) && (!wrapData))
		throw std::out_of_range("Too few values to assign to the tensor.");

	std::vector<size_t> treePath, maxPath;
	for(const auto& [lower, upper] : indices)
	{
		treePath.emplace_back(lower);
		maxPath.emplace_back(upper);
	}

	// for quicker iterating over data array when last dimensions are left with unspecified indices
	size_t wholeDimensionsOffset = 1;
	for(size_t i = shape_.size() - 1; i > treePath.size() - 1; i--)
		wholeDimensionsOffset *= shape_[i];

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
				dataIter = newData.begin();
		}

		for(size_t i = treePath.size() - 1; i < treePath.size(); i--)
		{
			treePath[i]++;
			if(treePath[i] < maxPath[i])
				break;

			treePath[i] = (indices.begin() + i)->first;
		}
	}
}

template <typename valueType>
void BasicTensor<valueType>::fill(const ITensorInitializer<valueType>&& initializer)
{
	size_t elementPos = 0;
	while((initializer.canYield()) && (elementPos < length_))
	{
		data_[elementPos] = initializer.yield();
		elementPos++;
	}
	if(elementPos < length_)
		throw std::out_of_range("Too few values to assign to the tensor.");
}

} // namespace mlCore
