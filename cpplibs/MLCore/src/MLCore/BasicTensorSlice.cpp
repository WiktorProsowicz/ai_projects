#include "MLCore/BasicTensorSlice.h"

#include <iomanip>

#include "MLCore/BasicTensor.h"

#define OPERATION_WITH_SCALAR_RHS(op, rhs)                                                                   \
	const auto chunkLength = _computeChunkLength();                                                          \
	for(auto* dataPtr : *dataChunks_)                                                                        \
	{                                                                                                        \
		for(size_t dataIdx = 0; dataIdx < chunkLength; dataIdx++)                                            \
		{                                                                                                    \
			dataPtr[dataIdx] op rhs;                                                                         \
		}                                                                                                    \
	}

#define OPERATION_WITH_ARRAY_RHS(op)                                                                         \
	const auto chunkLength = _computeChunkLength();                                                          \
	const auto sliceSize = _computeSliceSize();                                                              \
                                                                                                             \
	if((sliceSize < data.size()) || (sliceSize % data.size()) != 0)                                          \
	{                                                                                                        \
		throw std::invalid_argument("Cannot align provided elements with the slice's spanned data.");        \
	}                                                                                                        \
                                                                                                             \
	auto dataPtr = data.data();                                                                              \
	size_t dataIdx = 0;                                                                                      \
                                                                                                             \
	for(auto* tensorDataPtr : *dataChunks_)                                                                  \
	{                                                                                                        \
		for(size_t chunkIdx = 0; chunkIdx < chunkLength; ++chunkIdx, dataIdx = (dataIdx + 1) % data.size())  \
		{                                                                                                    \
			tensorDataPtr[chunkIdx] op dataPtr[dataIdx];                                                     \
		}                                                                                                    \
	}

namespace
{
/**
 * @brief Computes index of the element that separates the `shape` so that the right side is fully covered by
 * the `indices` and the left one not. The function assumes that the `indices` have been validated.
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
size_t getPivotShapeElement(const std::vector<size_t>& shape,
							const std::vector<std::pair<size_t, size_t>>& indices)
{
	int64_t shapeIndex = shape.size() - 1;

	for(; shapeIndex >= 0; shapeIndex--)
	{
		if((indices.at(shapeIndex).first != 0) || (indices.at(shapeIndex).second != shape.at(shapeIndex)))
		{
			break;
		}
	}

	return shapeIndex + 1;
}
} // namespace

namespace mlCore
{
// Explicit template class instantiation
template class BasicTensorSlice<double>;

// Explicit template function instantiation
template std::ostream& operator<<(std::ostream& os, const BasicTensorSlice<double>& slice);

template <typename ValueType>
BasicTensorSlice<ValueType>::BasicTensorSlice(const BasicTensorSlice<ValueType>& other)
	: tensor_(other.tensor_)
	, indices_(other.indices_)
	, dataChunks_(std::make_unique<std::vector<ValueType*>>(*other.dataChunks_))
{}

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
	, dataChunks_(std::make_unique<std::vector<ValueType*>>(_computeDataPointers()))
{}

template <typename ValueType>
std::vector<size_t> BasicTensorSlice<ValueType>::_computeSliceShape() const
{
	std::vector<size_t> collectedShapeDims(indices_.size(), 0);

	std::transform(indices_.cbegin(),
				   indices_.cend(),
				   collectedShapeDims.begin(),
				   [](const auto& indexPair) { return indexPair.second - indexPair.first; });

	return collectedShapeDims;
}

template <typename ValueType>
std::vector<ValueType*> BasicTensorSlice<ValueType>::_computeDataPointers() const
{
	std::vector<ValueType*> dataPointers;

	const auto& tShape = tensor_.get().shape_;
	const auto pivotElement = getPivotShapeElement(tShape, indices_);

	std::function<std::vector<ValueType*>(ValueType*, size_t, size_t)> recurseGetPointers;

	recurseGetPointers = [&tShape, &pivotElement, &recurseGetPointers, this](
							 ValueType* data, size_t shapeIdx, size_t currentSpan)
	{
		if(shapeIdx == pivotElement)
		{
			return std::vector<ValueType*>{data};
		}

		const auto nextSpan = currentSpan / tShape.at(shapeIdx);

		std::vector<ValueType*> pointers;

		for(size_t nextSpanIndex = indices_.at(shapeIdx).first; nextSpanIndex < indices_.at(shapeIdx).second;
			nextSpanIndex++)
		{
			const auto collectedPointers =
				recurseGetPointers(data + nextSpanIndex * nextSpan, shapeIdx + 1, nextSpan);

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
size_t BasicTensorSlice<ValueType>::_computeSliceSize() const
{
	const auto shape = _computeSliceShape();

	return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::_assign(const std::span<const ValueType>& data)
{
	if(data.size() == 1)
	{
		OPERATION_WITH_SCALAR_RHS(=, data[0]);
	}
	else
	{
		OPERATION_WITH_ARRAY_RHS(=);
	}
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::_assignAdd(const std::span<ValueType>& data)
{
	if(data.size() == 1)
	{
		OPERATION_WITH_SCALAR_RHS(+=, data[0]);
	}
	else
	{
		OPERATION_WITH_ARRAY_RHS(+=);
	}
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::_assignSubtract(const std::span<ValueType>& data)
{
	if(data.size() == 1)
	{
		OPERATION_WITH_SCALAR_RHS(-=, data[0]);
	}
	else
	{
		OPERATION_WITH_ARRAY_RHS(-=);
	}
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::_assignMultiply(const std::span<ValueType>& data)
{
	if(data.size() == 1)
	{
		OPERATION_WITH_SCALAR_RHS(*=, data[0]);
	}
	else
	{
		OPERATION_WITH_ARRAY_RHS(*=);
	}
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::_assignDivide(const std::span<ValueType>& data)
{
	if(data.size() == 1)
	{
		OPERATION_WITH_SCALAR_RHS(/=, data[0]);
	}
	else
	{
		OPERATION_WITH_ARRAY_RHS(/=);
	}
}

#define OPERATION_WITH_SLICE_AS_RHS(op)                                                                      \
	const auto zippedDataPointers = _determineBroadcastedDataPointers(other);                                \
                                                                                                             \
	const auto oChunkLength = other._computeChunkLength();                                                   \
	const auto tChunkLength = _computeChunkLength();                                                         \
                                                                                                             \
	for(const auto& [thisDataPtr, otherDataPtr] : zippedDataPointers)                                        \
	{                                                                                                        \
		if(oChunkLength == tChunkLength)                                                                     \
		{                                                                                                    \
			for(size_t dataIdx = 0; dataIdx < tChunkLength; dataIdx++)                                       \
			{                                                                                                \
				thisDataPtr[dataIdx] op otherDataPtr[dataIdx];                                               \
			}                                                                                                \
		}                                                                                                    \
		else                                                                                                 \
		{                                                                                                    \
			for(size_t dataIdx = 0; dataIdx < tChunkLength; dataIdx++)                                       \
			{                                                                                                \
				thisDataPtr[dataIdx] op* otherDataPtr;                                                       \
			}                                                                                                \
		}                                                                                                    \
	}

template <typename ValueType>
void BasicTensorSlice<ValueType>::assign(const BasicTensorSlice& other)
{
	OPERATION_WITH_SLICE_AS_RHS(=);
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::assignAdd(const BasicTensorSlice& other)
{
	OPERATION_WITH_SLICE_AS_RHS(+=);
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::assignSubtract(const BasicTensorSlice& other)
{
	OPERATION_WITH_SLICE_AS_RHS(-=);
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::assignMultiply(const BasicTensorSlice& other)
{
	OPERATION_WITH_SLICE_AS_RHS(*=);
}

template <typename ValueType>
void BasicTensorSlice<ValueType>::assignDivide(const BasicTensorSlice& other)
{
	OPERATION_WITH_SLICE_AS_RHS(/=);
}

namespace
{
/// Tells whether the `shapeFrom` can be broadcasted to the `shapeTo`.
bool isShapeBroadcastable(const std::vector<size_t>& shapeFrom, const std::vector<size_t>& shapeTo)
{
	if(shapeTo.size() != shapeFrom.size())
	{
		return false;
	}

	for(size_t shapeIdx = 0; shapeIdx < shapeFrom.size(); shapeIdx++)
	{
		if((shapeFrom.at(shapeIdx) != shapeTo.at(shapeIdx)) && (shapeFrom.at(shapeIdx) != 1))
		{
			return false;
		}
	}

	return true;
}

/// Modifies the `shape` so that the indices after the `pivotElement` are merged into a single dimension.
std::vector<size_t>
mergeShape(const std::vector<size_t> shape, const size_t pivotElement, const size_t chunkLength)
{
	if(pivotElement == shape.size())
	{
		return shape;
	}

	std::vector<size_t> mergedShape;

	for(size_t shapeIdx = 0; shapeIdx < pivotElement; shapeIdx++)
	{
		mergedShape.push_back(shape.at(shapeIdx));
	}

	mergedShape.push_back(chunkLength);

	return mergedShape;
}

std::vector<size_t> truncateShape(const std::vector<size_t>& shape, const size_t pivotElement)
{
	std::vector<size_t> truncatedShape;

	for(size_t shapeIdx = 0; shapeIdx < pivotElement; shapeIdx++)
	{
		truncatedShape.push_back(shape.at(shapeIdx));
	}

	return truncatedShape;
}

size_t getFlattenedIndex(const std::vector<size_t>& shape, const std::vector<size_t>& indices)
{
	size_t offset = 1;
	size_t flattenedIndex = 0;

	for(int64_t shapeIdx = shape.size() - 1; shapeIdx >= 0; shapeIdx--)
	{
		flattenedIndex += indices.at(shapeIdx) * offset;
		offset *= shape.at(shapeIdx);
	}

	return flattenedIndex;
}

size_t computeNElementsInShape(const std::span<const size_t>& shape)
{
	return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}
} // namespace

template <typename ValueType>
SlicedTensorIterator<ValueType> BasicTensorSlice<ValueType>::begin() const
{
	std::vector<size_t> firstElementPath(indices_.size());

	std::transform(indices_.cbegin(),
				   indices_.cend(),
				   firstElementPath.begin(),
				   [](const auto& indices) { return indices.first; });

	auto* startDataPtr = tensor_.get().data_ + getFlattenedIndex(tensor_.get().shape_, firstElementPath);

	return SlicedTensorIterator<ValueType>(startDataPtr, *dataChunks_, _computeChunkLength(), 0);
}

template <typename ValueType>
SlicedTensorIterator<ValueType> BasicTensorSlice<ValueType>::end() const
{
	std::vector<size_t> lastElementPath(indices_.size());

	std::transform(indices_.cbegin(),
				   indices_.cend(),
				   lastElementPath.begin(),
				   [](const auto& indices) { return indices.second - 1; });

	auto* endDataPtr = tensor_.get().data_ + getFlattenedIndex(tensor_.get().shape_, lastElementPath) + 1;

	return SlicedTensorIterator<ValueType>(
		endDataPtr, *dataChunks_, _computeChunkLength(), (dataChunks_->size() * _computeChunkLength()));
}

template <typename ValueType>
std::vector<std::pair<ValueType*, ValueType*>>
BasicTensorSlice<ValueType>::_determineBroadcastedDataPointers(const BasicTensorSlice& other) const
{
	const auto& tShape = tensor_.get().shape_;
	const auto& oShape = other.tensor_.get().shape_;

	const auto pivotElement = getPivotShapeElement(tShape, indices_);
	const auto oPivotElement = getPivotShapeElement(oShape, other.indices_);

	const auto mergedShapeThis = mergeShape(_computeSliceShape(), pivotElement, _computeChunkLength());
	const auto mergedShapeOther =
		mergeShape(other._computeSliceShape(), oPivotElement, other._computeChunkLength());

	if(!isShapeBroadcastable(mergedShapeOther, mergedShapeThis))
	{
		throw std::invalid_argument("Unable to broadcast the rhs slice.");
	}

	const auto truncatedShapeThis = truncateShape(mergedShapeThis, pivotElement);

	std::vector<std::pair<ValueType*, ValueType*>> zippedDataPointers;

	const auto& dataPointersThis = *dataChunks_;
	const auto& dataPointersOther = *(other.dataChunks_);

	std::vector<size_t> indicesPathThis(mergedShapeThis.size(), 0);
	std::vector<size_t> indicesPathOther(mergedShapeOther.size(), 0);

	for(size_t iteration = 0;
		iteration < computeNElementsInShape(std::span(truncatedShapeThis.begin(), truncatedShapeThis.size()));
		iteration++)
	{
		const auto flattenedIndexThis = getFlattenedIndex(mergedShapeThis, indicesPathThis);
		const auto flattenedIndexOther = getFlattenedIndex(mergedShapeOther, indicesPathOther);

		zippedDataPointers.push_back(
			{dataPointersThis.at(flattenedIndexThis), dataPointersOther.at(flattenedIndexOther)});

		for(int64_t shapeIdx = indicesPathThis.size() - 1; shapeIdx >= 0; shapeIdx--)
		{
			indicesPathThis.at(shapeIdx)++;

			if((mergedShapeOther.at(shapeIdx) != 1) &&
			   (static_cast<uint64_t>(shapeIdx) != (indicesPathThis.size() - 1)))
			{
				indicesPathOther.at(shapeIdx)++;
			}

			if(indicesPathThis.at(shapeIdx) < mergedShapeThis.at(shapeIdx))
			{
				break;
			}
			indicesPathThis.at(shapeIdx) = 0;
			indicesPathOther.at(shapeIdx) = 0;
		}
	}

	return zippedDataPointers;
}

namespace
{
template <typename ValueType>
void serializeContiguousMemory(std::ostream& ostream,
							   const std::span<ValueType>& data,
							   const std::string& preamble,
							   const std::span<size_t>& shape,
							   const int blockLength)
{
	std::function<void(std::span<size_t>::iterator,
					   typename std::span<ValueType>::iterator,
					   typename std::span<ValueType>::iterator,
					   const std::string&)>
		recurseSerialize;

	recurseSerialize =
		[&recurseSerialize, &ostream, &shape, blockLength](const std::span<size_t>::iterator shapeIter,
														   typename std::span<ValueType>::iterator dataBeg,
														   typename std::span<ValueType>::iterator dataEnd,
														   const std::string& preamble)
	{
		if(shapeIter == std::prev(shape.end()))
		{
			ostream << "\n" << preamble << "[";

			for(auto dataIter = dataBeg; dataIter != dataEnd - 1; dataIter++)
			{
				ostream << std::setw(blockLength) << *dataIter << ", ";
			}

			ostream << std::setw(blockLength) << *(dataEnd - 1);

			ostream << "]";
		}
		else
		{
			const auto offset = std::distance(dataBeg, dataEnd) / *shapeIter;

			ostream << "\n" << preamble << "[";

			for(size_t dimIdx = 0; dimIdx < *shapeIter; dimIdx++)
			{
				recurseSerialize(shapeIter + 1,
								 dataBeg + (dimIdx * offset),
								 dataBeg + ((dimIdx + 1) * offset),
								 preamble + " ");
			}

			ostream << "\n" << preamble << "]";
		}
	};

	recurseSerialize(shape.begin(), data.begin(), data.end(), preamble);
}

template <typename ValueType>
int getBlockSize(const std::vector<ValueType*>& dataPtrs, const size_t& chunkLength)
{
	return std::accumulate(dataPtrs.cbegin(),
						   dataPtrs.cend(),
						   int{0},
						   [chunkLength](const size_t& acc, const ValueType* dataPtr)
						   {
							   const auto* const maxForChunk =
								   std::max_element(dataPtr,
													dataPtr + chunkLength,
													[](const ValueType& lhs, const ValueType& rhs) {
														return (std::ostringstream{} << lhs).str().size() <
															   (std::ostringstream{} << rhs).str().size();
													});

							   const auto maxSizeForChunk =
								   (std::ostringstream{} << *maxForChunk).str().size();

							   return std::max(acc, maxSizeForChunk);
						   });
}
} // namespace

template <typename SliceValueType>
std::ostream& operator<<(std::ostream& ostream, const BasicTensorSlice<SliceValueType>& slice)
{
	ostream << "<BasicTensorSlice dtype=" << typeid(SliceValueType).name()
			<< " shape=" << stringifyVector(slice._computeSliceShape()) << ">";

	const auto& dataPtrs = *(slice.dataChunks_);
	auto tShape = slice._computeSliceShape();
	const auto chunkLength = slice._computeChunkLength();
	const auto blockLength = getBlockSize(dataPtrs, chunkLength);

	auto mergedShape =
		mergeShape(tShape, getPivotShapeElement(slice.tensor_.get().shape(), slice.indices_), chunkLength);

	std::function<void(std::vector<size_t>::iterator,
					   typename std::vector<SliceValueType*>::const_iterator,
					   typename std::vector<SliceValueType*>::const_iterator,
					   const std::string&)>
		recursePrint;

	recursePrint = [&recursePrint, &ostream, &tShape, &mergedShape, &chunkLength, &blockLength](
					   std::vector<size_t>::iterator shapeIter,
					   typename std::vector<SliceValueType*>::const_iterator dataBeg,
					   typename std::vector<SliceValueType*>::const_iterator dataEnd,
					   const std::string& preamble)
	{
		if(shapeIter == std::prev(mergedShape.end()))
		{
			serializeContiguousMemory(
				ostream,
				std::span(*dataBeg, chunkLength),
				preamble,
				std::span(tShape.begin() + mergedShape.size() - 1, tShape.size() - mergedShape.size() + 1),
				blockLength);
		}
		else
		{
			const auto offset = std::distance(dataBeg, dataEnd) / *shapeIter;

			ostream << "\n" << preamble << "[";

			for(size_t dimIdx = 0; dimIdx < *shapeIter; dimIdx++)
			{
				recursePrint(shapeIter + 1,
							 dataBeg + (dimIdx * offset),
							 dataBeg + ((dimIdx + 1) * offset),
							 preamble + " ");
			}

			ostream << "\n" << preamble << "]";
		}
	};

	recursePrint(mergedShape.begin(), dataPtrs.cbegin(), dataPtrs.cend(), "");

	return ostream;
}

} // namespace mlCore
