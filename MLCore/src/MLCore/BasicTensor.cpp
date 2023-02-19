
#include "MLCore/BasicTensor.h"
#include "LoggingLib/LoggingLib.h"
#include <algorithm>
#include <exception>
#include <numeric>

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
 * Throws std::out_of_range if max[i] > shape[i] or 0 > indices.size() > shape_.size()
 * 
 * @tparam valueType dtype of tensor
 * @param indices list of pairs of min-max indices from axis zero i.e for tensor([[1, 2], [3, 4]]) -> list{{0, 1}} -> [1, 2]
 */
template <typename valueType>
void BasicTensor<valueType>::checkIndicesList_(
	const std::initializer_list<std::pair<size_t, size_t>>& indices) const
{
	if(!indices.size())
		throw std::out_of_range("Indices list must have minimum length of 1.");

	if(indices.size() > shape_.size())
		throw std::out_of_range("Indices list cannot be longer than tensor's shape.");

	for(auto [it, end, i] = std::tuple{indices.begin(), indices.end(), 0}; it < end; ++it, ++i)
	{
		const auto& [lower, upper] = *it;
		if(upper <= lower)
			throw std::out_of_range(
				std::string("Upper index is not greater than lower for shape '") +
				stringifyVector(shape_) + "' at index " + std::to_string(i) + ".");

		if(upper >= shape_[i])
			throw std::out_of_range(
				std::string(
					"Upper index cannot be greater than particular dimension size for shape '") +
				stringifyVector(shape_) + "' at index " + std::to_string(i) + ".");
	}
}

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::operator*(const BasicTensor& other) const
{
	return performOperation_(other, mulOperator_);
}

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::operator-(const BasicTensor& other) const
{
	return performOperation_(other, minusOperator_);
}

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::operator+(const BasicTensor& other) const
{
	return performOperation_(other, plusOperator_);
}

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::operator/(const BasicTensor& other) const
{
	return performOperation_(other, divOperator_);
}

template <typename valueType>
BasicTensor<valueType> BasicTensor<valueType>::performOperation_(
	const BasicTensor<valueType>& other,
	const std::function<valueType(const valueType*, const valueType*)>& op_) const
{
	if(shape_ == other.shape_)
	{
		BasicTensor<valueType> ret(shape_);
		for(size_t i = 0; i < length_; i++)
			ret.data_[i] = op_(data_ + i, other.data_ + i);

		return ret;
	}

	// checking if the rules of broadcasting are not breached
	for(size_t i = 0; i < std::min(shape_.size(), other.shape_.size()); i++)
	{
		if((shape_[i] != 1) && (other.shape_[i] != 1) && (shape_[i] != other.shape_[i]))
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

	uint8_t initVal = 0;
	if((&op_ == &mulOperator_) || (&op_ == &divOperator_))
		initVal = 1;

	BasicTensor<valueType> ret(retShape, initVal);

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
		for(size_t i = path.size() - 1; i >= 0; i--)
		{
			position += offset * path[i];
			offset *= shape[i];
		}

		return position;
	};

	// updates tree paths untill all elements are processed and constantly assigns proper elements to proper places
	std::function<void(valueType* const, const valueType* const, 
					   const std::vector<size_t>&, 
					   std::function<valueType(const valueType*, const valueType*)>)> appendTensor;

	appendTensor = [&factorTreePath, &destTreePath, &retShape, &computeElementPosition](valueType* const destDataPtr,
															   const valueType* const factorDataPtr,
															   const std::vector<size_t>& factorShape,
															   std::function<valueType(const valueType*, const valueType*)> op)
	{
		size_t elementsProcessed = 0;
		size_t destTensorLength = std::accumulate(retShape.cbegin(), retShape.cend(), 0, 
													[](const auto curr, const auto dim){ return curr * dim; });

		while(elementsProcessed < destTensorLength)
		{

			for(size_t i = retShape.size() - 1; i >= 0; i--) {

				const auto destElemPos = computeElementPosition(destTreePath, retShape);
				const auto factorElemPos = computeElementPosition(factorTreePath, factorShape);

				destDataPtr[destElemPos] = op(destDataPtr + destElemPos, factorDataPtr + factorElemPos);

				elementsProcessed++;
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

	std::function<valueType(const valueType*, const valueType*)> firstOp;
	if((&op_ == &plusOperator_) || (&op_ == &minusOperator_))
		firstOp = plusOperator_;
	else
		firstOp = mulOperator_;

	appendTensor(ret.data_, this->data_, this->shape_, firstOp);

	for(size_t i = 0; i < biggerSize; i++)
	{
		destTreePath[i] = 0;
		factorTreePath[i] = 0;
	}

	appendTensor(ret.data_, other.data_, other.shape_, op_);

	return ret;
}

template <typename valueType>
void BasicTensor<valueType>::assign(std::initializer_list<std::pair<size_t, size_t>> indices,
									std::initializer_list<valueType> newData,
									const bool wrapData)
{
	checkIndicesList_(indices);
}

template <typename valueType>
void BasicTensor<valueType>::fill(std::shared_ptr<ITensorInitializer<valueType>> initializer)
{
	size_t elementPos = 0;
	while((initializer->canYield()) && (elementPos < length_))
	{
		data_[elementPos] = initializer->yield();
		elementPos++;
	}
	if(elementPos < length_)
		throw std::out_of_range("Too few values to assign to the tensor.");
}

} // namespace mlCore
