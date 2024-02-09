// __Related headers__
#include <MLCore/BasicTensorSlice.h>

// __Own software headers__
#include <MLCore/BasicTensor.h>

#define OPERATION_WITH_SCALAR_RHS(op, rhs)                                                                                       \
	const auto chunkLength = _computeChunkLength();                                                                              \
	for(auto* dataPtr : _computeDataPointers())                                                                                  \
	{                                                                                                                            \
		for(size_t dataIdx = 0; dataIdx < chunkLength; dataIdx++)                                                                \
		{                                                                                                                        \
			dataPtr[dataIdx] op rhs;                                                                                             \
		}                                                                                                                        \
	}

namespace
{
/**
 * @brief Computes index of the element that separates the `shape` so that the right side is fully covered by the `indices` and the left one not.
 * The function assumes that the `indices` have been validated.
 * 
 * @example
 * 	shape: {4, 5, 6, 7, 8}
 * 	indices: {(2, 3), (3, 4), (0, 6), (0, 7)}
 * 	Result: 2 
 * 
 * @param shape Shape to divide.
 * @param indices Indices referring to the spanned part of the shape.
 * @return Index of the leftmost shape's element spanned entirely by the respective indices.
 * If there's no such element, the index == len(size) is returned.
 */
size_t getPivotShapeElement(const std::vector<size_t>& shape, const std::vector<std::pair<size_t, size_t>>& indices)
{
	int64_t shapeIndex = shape.size() - 1;

	for(; shapeIndex >= 0; shapeIndex--)
	{
		if(static_cast<uint64_t>(shapeIndex) >= indices.size())
		{
			continue;
		}

		if((indices.at(shapeIndex).first == 0) && (indices.at(shapeIndex).second == shape.at(shapeIndex)))
		{
			continue;
		}
	}

	return shapeIndex + 1;
}
} // namespace

namespace mlCore
{

// Explicit template class instantiation
template class BasicTensorSlice<double>;

template <typename ValueType>
BasicTensorSlice<ValueType>::BasicTensorSlice(const BasicTensorSlice<ValueType>& other)
	: tensor_(other.tensor_)
	, indices_(other.indices_)
{ }

template <typename ValueType>
BasicTensorSlice<ValueType>& BasicTensorSlice<ValueType>::operator=(const BasicTensorSlice<ValueType>& other)
{
	if(&other != this)
	{
		tensor_ = other.tensor_;
		indices_ = other.indices_;
	}

	return *this;
}

template <typename ValueType>
BasicTensorSlice<ValueType>::BasicTensorSlice(BasicTensor<ValueType>& associatedTensor,
											  const std::vector<std::pair<size_t, size_t>>& indices)
	: tensor_(associatedTensor)
	, indices_(indices)
{ }

template <typename ValueType>
std::vector<size_t> BasicTensorSlice<ValueType>::_computeSliceShape() const
{
	const auto pivotElement = getPivotShapeElement(tensor_.get().shape_, indices_);

	std::vector<size_t> collectedShapeDims;

	for(size_t indicesIdx = 0; indicesIdx < pivotElement; indicesIdx++)
	{
		const auto indicesDifference = indices_.at(indicesIdx).second - indices_.at(indicesIdx).first;

		if(indicesDifference != 0)
		{
			collectedShapeDims.push_back(indicesDifference);
		}
	}

	for(size_t shapeIdx = pivotElement; shapeIdx < tensor_.get().shape_.size(); shapeIdx++)
	{
		collectedShapeDims.push_back(tensor_.get().shape_.at(shapeIdx));
	}

	return collectedShapeDims;
}

template <typename ValueType>
std::vector<ValueType*> BasicTensorSlice<ValueType>::_computeDataPointers() const
{
	std::vector<ValueType*> dataPointers;

	const auto& tShape = tensor_.get().shape_;
	const auto pivotElement = getPivotShapeElement(tShape, indices_);

	std::function<std::vector<ValueType*>(ValueType*, size_t, size_t)> recurseGetPointers;

	recurseGetPointers =
		[&tShape, &pivotElement, &recurseGetPointers, this](ValueType* data, size_t shapeIdx, size_t currentSpan) {
			if(shapeIdx == pivotElement)
			{
				return std::vector<ValueType*>{data};
			}

			const auto nextSpan = currentSpan / tShape.at(shapeIdx);

			std::vector<ValueType*> pointers;

			for(size_t nextSpanIndex = indices_.at(shapeIdx).first; nextSpanIndex < indices_.at(shapeIdx).second; nextSpanIndex++)
			{
				const auto collectedPointers = recurseGetPointers(data + nextSpanIndex * nextSpan, shapeIdx + 1, nextSpan);

				std::copy(collectedPointers.cbegin(), collectedPointers.cend(), std::back_inserter(pointers));
			}

			return pointers;
		};

	return recurseGetPointers(tensor_.get().data_, 0, tensor_.get().length_);
}

template <typename ValueType>
size_t BasicTensorSlice<ValueType>::_computeChunkLength() const
{
	size_t chunkLength = 1;

	const auto& tShape = tensor_.get().shape_;

	for(size_t shapeIdx = getPivotShapeElement(tShape, indices_); shapeIdx < tShape.size(); shapeIdx++)
	{
		chunkLength *= tShape.at(shapeIdx);
	}

	return chunkLength;
}
template <typename ValueType>
void BasicTensorSlice<ValueType>::assign(ValueType value)
{
	OPERATION_WITH_SCALAR_RHS(=, value);
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::assignAdd(ValueType value)
{
	OPERATION_WITH_SCALAR_RHS(+=, value);
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::assignSubtract(ValueType value)
{
	OPERATION_WITH_SCALAR_RHS(-=, value);
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::assignDivide(ValueType value)
{
	OPERATION_WITH_SCALAR_RHS(/=, value);
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::assignMultiply(ValueType value)
{
	OPERATION_WITH_SCALAR_RHS(*=, value);
}

} // namespace mlCore