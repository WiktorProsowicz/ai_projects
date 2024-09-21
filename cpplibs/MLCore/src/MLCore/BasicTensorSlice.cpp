#include "MLCore/BasicTensorSlice.h"

#include <cstddef>
#include <functional>
#include <iomanip>
#include <iterator>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "MLCore/BasicTensor.h"
#include "MLCore/SlicedTensorIterator.hpp"
#include "MLCore/UtilitiesImpl.h"

#define OPERATION_WITH_SCALAR_RHS(op, rhs)                                                                   \
	const auto chunkLength = _computeChunkLength();                                                          \
	for(auto* dataPtr : *_dataChunks)                                                                        \
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
	for(auto* tensorDataPtr : *_dataChunks)                                                                  \
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
 * 	indices: {(2, 3), (3, 4), (2, 5), (0, 7), (0, 8)}
 * 	Result: 3
 *
 * @param shape Shape to divide.
 * @param indices Indices referring to the spanned part of the shape.
 * @return Index of the leftmost shape's element spanned entirely by the respective indices.
 * If there's no such element, the index == len(size) is returned.
 */
size_t getPivotShapeElement(const std::vector<size_t>& shape,
							const std::vector<std::pair<size_t, size_t>>& indices)
{

	for(auto [shapeIt, indicesIt] = std::tuple{shape.crbegin(), indices.crbegin()};
		(shapeIt < shape.crend()) && (indicesIt < indices.crend());
		shapeIt++, indicesIt++)
	{
		if((indicesIt->first != 0) || (indicesIt->second != *shapeIt))
		{
			return static_cast<size_t>(std::distance(shapeIt, shape.crend()));
		}
	}

	return 0;
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
	: _tensor(other._tensor)
	, _indices(other._indices)
	, _dataChunks(std::make_unique<std::vector<ValueType*>>(*other._dataChunks))
{}

template <typename ValueType>
BasicTensorSlice<ValueType>& BasicTensorSlice<ValueType>::operator=(const BasicTensorSlice<ValueType>& other)
{
	if(&other != this)
	{
		_tensor = other._tensor;
		_indices = other._indices;
	}

	return *this;
}

template <typename ValueType>
BasicTensorSlice<ValueType>::BasicTensorSlice(BasicTensor<ValueType>& associatedTensor,
											  const std::vector<std::pair<size_t, size_t>>& indices)
	: _tensor(associatedTensor)
	, _indices(indices)
	, _dataChunks(std::make_unique<std::vector<ValueType*>>(_computeDataPointers()))
{}

template <typename ValueType>
std::vector<size_t> BasicTensorSlice<ValueType>::_computeSliceShape() const
{
	std::vector<size_t> collectedShapeDims(_indices.size(), 0);

	std::transform(_indices.cbegin(),
				   _indices.cend(),
				   collectedShapeDims.begin(),
				   [](const auto& indexPair) { return indexPair.second - indexPair.first; });

	return collectedShapeDims;
}

template <typename ValueType>
std::vector<ValueType*> BasicTensorSlice<ValueType>::_computeDataPointers() const
{
	std::vector<ValueType*> dataPointers;

	const auto& tShape = _tensor.get()._shape;
	const auto pivotElement = getPivotShapeElement(tShape, _indices);

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

		for(size_t nextSpanIndex = _indices.at(shapeIdx).first; nextSpanIndex < _indices.at(shapeIdx).second;
			nextSpanIndex++)
		{
			const auto collectedPointers =
				recurseGetPointers(data + nextSpanIndex * nextSpan, shapeIdx + 1, nextSpan);

			std::copy(collectedPointers.cbegin(), collectedPointers.cend(), std::back_inserter(pointers));
		}

		return pointers;
	};

	return recurseGetPointers(_tensor.get()._data, 0, _tensor.get()._length);
}

template <typename ValueType>
size_t BasicTensorSlice<ValueType>::_computeChunkLength() const
{
	size_t chunkLength = 1;

	const auto& tShape = _tensor.get()._shape;

	for(size_t shapeIdx = getPivotShapeElement(tShape, _indices); shapeIdx < tShape.size(); shapeIdx++)
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
mergeShape(const std::vector<size_t>& shape, const size_t pivotElement, const size_t chunkLength)
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

/// Returns position of an element specified by `indices` within data organized by the `shape`.
size_t getFlattenedIndex(const std::vector<size_t>& shape, const std::vector<size_t>& indices)
{
	size_t offset = std::accumulate(shape.cbegin(), shape.cend(), size_t{1}, std::multiplies<>{});

	auto shapeIt = shape.cbegin();

	return std::accumulate(indices.cbegin(),
						   indices.cend(),
						   size_t{0},
						   [&offset, &shapeIt](const auto& curr, const auto& index)
						   { return curr + (offset /= *(shapeIt++)) * index; });
}

size_t computeNElementsInShape(const std::span<const size_t>& shape)
{
	return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
}
} // namespace

template <typename ValueType>
SlicedTensorIterator<ValueType> BasicTensorSlice<ValueType>::begin() const
{
	std::vector<size_t> firstElementPath(_indices.size());

	std::transform(_indices.cbegin(),
				   _indices.cend(),
				   firstElementPath.begin(),
				   [](const auto& indices) { return indices.first; });

	auto* startDataPtr = _tensor.get()._data + getFlattenedIndex(_tensor.get()._shape, firstElementPath);

	return SlicedTensorIterator<ValueType>(startDataPtr, *_dataChunks, _computeChunkLength(), 0);
}

template <typename ValueType>
SlicedTensorIterator<ValueType> BasicTensorSlice<ValueType>::end() const
{
	std::vector<size_t> lastElementPath(_indices.size());

	std::transform(_indices.cbegin(),
				   _indices.cend(),
				   lastElementPath.begin(),
				   [](const auto& indices) { return indices.second - 1; });

	auto* endDataPtr = _tensor.get()._data + getFlattenedIndex(_tensor.get()._shape, lastElementPath) + 1;

	return SlicedTensorIterator<ValueType>(
		endDataPtr, *_dataChunks, _computeChunkLength(), (_dataChunks->size() * _computeChunkLength()));
}

template <typename ValueType>
std::vector<std::pair<ValueType*, ValueType*>>
BasicTensorSlice<ValueType>::_determineBroadcastedDataPointers(const BasicTensorSlice& other) const
{
	const auto& tShape = _tensor.get()._shape;
	const auto& oShape = other._tensor.get()._shape;

	const auto pivotElement = getPivotShapeElement(tShape, _indices);
	const auto oPivotElement = getPivotShapeElement(oShape, other._indices);

	const auto mergedShapeThis = mergeShape(_computeSliceShape(), pivotElement, _computeChunkLength());
	const auto mergedShapeOther =
		mergeShape(other._computeSliceShape(), oPivotElement, other._computeChunkLength());

	if(!isShapeBroadcastable(mergedShapeOther, mergedShapeThis))
	{
		throw std::invalid_argument("Unable to broadcast the rhs slice.");
	}

	const auto truncatedShapeThis = truncateShape(mergedShapeThis, pivotElement);

	std::vector<std::pair<ValueType*, ValueType*>> zippedDataPointers;

	const auto& dataPointersThis = *_dataChunks;
	const auto& dataPointersOther = *(other._dataChunks);

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

		auto [pathThisIt, pathOtherIt, shapeThisIt, shapeOtherIt] = std::tuple{indicesPathThis.rbegin(),
																			   indicesPathOther.rbegin(),
																			   mergedShapeThis.crbegin(),
																			   mergedShapeOther.crbegin()};

		for(; (pathThisIt < indicesPathThis.rend()) && (pathOtherIt < indicesPathOther.rend()) &&
			  (shapeThisIt < mergedShapeThis.crend()) && (shapeOtherIt < mergedShapeOther.crend());
			pathThisIt++, pathOtherIt++, shapeThisIt++, shapeOtherIt++)
		{
			(*pathThisIt)++;

			if((*shapeOtherIt != 1) && (shapeOtherIt != mergedShapeOther.crbegin()))
			{
				(*pathOtherIt)++;
			}

			if(*pathThisIt < *shapeThisIt)
			{
				break;
			}

			*pathThisIt = 0;
			*pathOtherIt = 0;
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
			const auto offset = static_cast<size_t>(std::distance(dataBeg, dataEnd)) / *shapeIter;

			ostream << "\n" << preamble << "[";

			for(size_t dimIdx = 0; dimIdx < *shapeIter; dimIdx++)
			{
				recurseSerialize(std::next(shapeIter),
								 std::next(dataBeg, static_cast<ptrdiff_t>(dimIdx * offset)),
								 std::next(dataBeg, static_cast<ptrdiff_t>((dimIdx + 1) * offset)),
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
						   0,
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
			<< " shape=" << detail::stringifyVector(slice._computeSliceShape()) << ">";

	const auto& dataPtrs = *(slice._dataChunks);
	auto tShape = slice._computeSliceShape();
	const auto chunkLength = slice._computeChunkLength();
	const auto blockLength = getBlockSize(dataPtrs, chunkLength);

	auto mergedShape =
		mergeShape(tShape, getPivotShapeElement(slice._tensor.get().shape(), slice._indices), chunkLength);

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
			const std::span<size_t> leftShape(
				std::next(tShape.begin(), static_cast<ptrdiff_t>(mergedShape.size() - 1)),
				tShape.size() - mergedShape.size() + 1);

			const std::span<SliceValueType> memory(*dataBeg, chunkLength);

			serializeContiguousMemory(ostream, memory, preamble, leftShape, blockLength);
		}
		else
		{
			const auto offset = static_cast<size_t>(std::distance(dataBeg, dataEnd)) / *shapeIter;

			ostream << "\n" << preamble << "[";

			for(size_t dimIdx = 0; dimIdx < *shapeIter; dimIdx++)
			{
				recursePrint(shapeIter + 1,
							 std::next(dataBeg, static_cast<ptrdiff_t>(dimIdx * offset)),
							 std::next(dataBeg, static_cast<ptrdiff_t>((dimIdx + 1) * offset)),
							 preamble + " ");
			}

			ostream << "\n" << preamble << "]";
		}
	};

	recursePrint(mergedShape.begin(), dataPtrs.cbegin(), dataPtrs.cend(), "");

	return ostream;
}

} // namespace mlCore
