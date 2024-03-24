#ifndef MLCORE_INCLUDE_MLCORE_BASICTENSORSLICE_H
#define MLCORE_INCLUDE_MLCORE_BASICTENSORSLICE_H

// __C++ standard headers__
#include <cstdio>
#include <functional>
#include <ostream>
#include <span>
#include <vector>

#include "MLCore/SlicedTensorIterator.hpp"

namespace mlCore
{
template <typename ValueType>
class BasicTensor;

namespace detail
{
template <typename Type>
concept SizedRange = requires(Type item) {
	requires std::ranges::range<Type>;
	item.size();
};

} // namespace detail

/**
 * @brief Represents part of the tensor taken by providing ranges of indices.
 *
 * Tensor slices are merely the views over mlCore::BasicTensor instances. There can be multiple
 * instances of slices created by providing different sets of indices as long as the referenced
 * tensor is alive, otherwise the slices are not valid. This can be compared to the dangling references
 * problem. No lifetime tracking is performed automatically.
 *
 * @tparam ValueType Underlying data type of the referenced tensor.
 */
template <typename ValueType>
class BasicTensorSlice
{
	template <typename BasicTensorValueType>
	friend class BasicTensor;

public:
	BasicTensorSlice() = delete; /// Default constructor.

	/**
	 * @brief Constructs tensor slice copying its configuration and linking it to the tensor associated do
	 * `other`.
	 *
	 * @param other Slice to copy.
	 *
	 */
	BasicTensorSlice(const BasicTensorSlice& other);

	BasicTensorSlice(BasicTensorSlice&&) = delete; /// Move constructor.

	/**
	 * @brief Copies another slice's configuration to this. Don't confuse it with `assign` method
	 * copying data referenced by other slice.
	 *
	 * @param other Slice to copy.
	 *
	 */
	BasicTensorSlice& operator=(const BasicTensorSlice& other);

	BasicTensorSlice& operator=(BasicTensorSlice&&) = delete; /// Move assignment.

	~BasicTensorSlice() = default; /// Default destructor.

	/**
	 * @brief Copies data from tensor referenced by `other` slice. Number of elements
	 * spanned by first slice should be divisible by the number of elements spanned by the second one.
	 *
	 * @param other A slice to copy data from the tensor referenced by.
	 */
	void assign(const BasicTensorSlice& other);

	/// @see BasicTensorSlice::assign(const BasicTensorSlice&)
	void assignAdd(const BasicTensorSlice& other);

	/// @see BasicTensorSlice::assign(const BasicTensorSlice&)
	void assignSubtract(const BasicTensorSlice& other);

	/// @see BasicTensorSlice::assign(const BasicTensorSlice&)
	void assignDivide(const BasicTensorSlice& other);

	/// @see BasicTensorSlice::assign(const BasicTensorSlice&)
	void assignMultiply(const BasicTensorSlice& other);

	template <detail::SizedRange Range>
	void assign(const Range& values)
	{
		_assign(std::span(values.begin(), values.size()));
	}

	template <detail::SizedRange Range>
	void assignAdd(const Range& values)
	{
		_assignAdd(std::span(values.begin(), values.size()));
	}

	template <detail::SizedRange Range>
	void assignSubtract(const Range& values)
	{
		_assignSubtract(std::span(values.begin(), values.size()));
	}

	template <detail::SizedRange Range>
	void assignDivide(const Range& values)
	{
		_assignDivide(std::span(values.begin(), values.size()));
	}

	template <detail::SizedRange Range>
	void assignMultiply(const Range& values)
	{
		_assignMultiply(std::span(values.begin(), values.size()));
	}

	/**
	 * @brief Returns iterator pointing to the beginning of the slice.
	 *
	 */
	SlicedTensorIterator<ValueType> begin() const;

	/**
	 * @brief Returns interator pointing to the end of the slice.
	 *
	 */
	SlicedTensorIterator<ValueType> end() const;

	template <typename SliceValueType>
	friend std::ostream& operator<<(std::ostream& ostream, const BasicTensorSlice<SliceValueType>& slice);

private:
	/**
	 * @brief Constructs the slice linking it to the tensor and providing referenced span with indices.
	 *
	 * @param associatedTensor Tensor to be referenced.
	 * @param indices Span indicating the shape of the slice. The length of the indices should be the same
	 * as the length of the source tensor's shape. Each pair of indices should be a range rather than a single
	 * index, i.e. (2, 3) instead of (2, 2).
	 */
	explicit BasicTensorSlice(BasicTensor<ValueType>& associatedTensor,
							  const std::vector<std::pair<size_t, size_t>>& indices);

	/**
	 * @brief Assigns the data from the span to the elements spanned by the slice.
	 *
	 * @param data View over the data to be assigned.
	 */
	void _assign(const std::span<const ValueType>& data);

	/// @see BasicTensorSlice::_assign(const std::span<ValueType>&)
	void _assignAdd(const std::span<ValueType>& data);

	/// @see BasicTensorSlice::_assign(const std::span<ValueType>&)
	void _assignSubtract(const std::span<ValueType>& data);

	/// @see BasicTensorSlice::_assign(const std::span<ValueType>&)
	void _assignDivide(const std::span<ValueType>& data);

	/// @see BasicTensorSlice::_assign(const std::span<ValueType>&)
	void _assignMultiply(const std::span<ValueType>& data);

	/// Computes the slice's shape based on its indices and the shape of the associated tensor.
	std::vector<size_t> _computeSliceShape() const;

	/// Computes an array of pointers to the places in memory where there are chunks of contiguous memory.
	std::vector<ValueType*> _computeDataPointers() const;

	/// Computes size of the chunk of contiguous memory spanned by the slice.
	size_t _computeChunkLength() const;

	/// Determines pairs of pointers to arrays of data that shall be aligned together after broadcasting.
	std::vector<std::pair<ValueType*, ValueType*>>
	_determineBroadcastedDataPointers(const BasicTensorSlice& other) const;

	/// Computes number off all data items spanned by the slice.
	size_t _computeSliceSize() const;

	/// Reference to the tensor spanned by the slice.
	std::reference_wrapper<mlCore::BasicTensor<ValueType>> tensor_;
	SliceIndices indices_;
	/// Contiguous data chunks spanned by the slice. Each chunk has size determined by the indices.
	std::unique_ptr<std::vector<ValueType*>> dataChunks_;
};

using TensorSlice = BasicTensorSlice<double>;

template <typename ValueType>
std::ostream& operator<<(std::ostream& ostream, const BasicTensorSlice<ValueType>& slice);

} // namespace mlCore

#endif
