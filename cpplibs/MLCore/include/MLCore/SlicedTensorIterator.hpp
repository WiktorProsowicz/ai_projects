#ifndef MLCORE_INCLUDE_MLCORE_SLICEDTENSORITERATOR_HPP
#define MLCORE_INCLUDE_MLCORE_SLICEDTENSORITERATOR_HPP

#include <iterator>
#include <memory>
#include "MLCore/Utilities.h"

namespace mlCore
{
/**
 * @brief Iterates over data spanned by a tensor slice according to provided
 *
 * @tparam ValueType
 */
template <typename ValueType>
class SlicedTensorIterator
{
public:
	template <typename T>
	friend class BasicTensorSlice;

	using iterator_category = std::random_access_iterator_tag;
	using value_type = ValueType;
	using difference_type = std::ptrdiff_t;
	using pointer = ValueType*;
	using reference = ValueType&;

	SlicedTensorIterator() = delete;

	SlicedTensorIterator(const SlicedTensorIterator&) = default;
	SlicedTensorIterator(SlicedTensorIterator&&) = delete;
	SlicedTensorIterator& operator=(const SlicedTensorIterator&) = default;
	SlicedTensorIterator& operator=(SlicedTensorIterator&&) = delete;

	~SlicedTensorIterator() = default;

	/// @brief Accesses underlying value.
	reference operator*() const
	{
		return *currentPtr_;
	}

	/// @brief Accesses underlying pointer.
	pointer operator->()
	{
		return currentPtr_;
	}

	/// @brief Increments the iterator and returns it.
	SlicedTensorIterator& operator++()
	{
		this->operator+=(1);
		return *this;
	}

	/// @brief Increments the iterator returning the untouched copy of it.
	SlicedTensorIterator operator++(int)
	{
		auto tmp = *this;
		++(*this);
		return tmp;
	}

	/// @brief Decrements the iterator and returns it.
	SlicedTensorIterator& operator--()
	{
		this->operator-=(1);
		return *this;
	}

	/// @brief Decrements the iterator returning the untouched copy of it.
	SlicedTensorIterator operator--(int)
	{
		auto tmp = *this;
		--(this);
		return tmp;
	}

	/// @brief Moves the iterator `n` times forwards and returns it.
	SlicedTensorIterator& operator-=(difference_type n)
	{
		offset_ -= n;

		if(!_isOffsetValid())
		{
			currentPtr_ -= n;
			return *this;
		}

		_updatePointer();

		return *this;
	}

	/// @brief Moves the iterator `n` times backwards and returns it.
	SlicedTensorIterator& operator+=(difference_type n)
	{
		offset_ += n;

		if(!_isOffsetValid())
		{
			currentPtr_ += n;
			return *this;
		}

		_updatePointer();

		return *this;
	}

	/// Tells whether the slice points to the same point in data as the other one.
	bool operator==(const SlicedTensorIterator& other) const
	{
		return currentPtr_ == other.currentPtr_;
	}

	/// Tells whether the slice points to the same point in data as the other one.
	bool operator!=(const SlicedTensorIterator& other) const
	{
		return currentPtr_ != other.currentPtr_;
	}

	/// Tells whether the slice comes first in order than the other one.
	bool operator<(const SlicedTensorIterator& other) const
	{
		return currentPtr_ < other.currentPtr_;
	}

private:
	/**
	 * @brief Creates a new iterator with parameters passed by a tensor slice.
	 *
	 * @param ptr Pointer to data spanned by the creating slice.
	 * @param dataChunks Chunks of contiguous data from the tensor.
	 * @param chunkLength Size of each chunk of data.
	 * @param offset Offset indicating the shift of the current pointer regarding the beginning
	 *  of the whole data spanned by the slice. The offset assumes the spanned data is contiguous.
	 */
	SlicedTensorIterator(pointer ptr,
						 const std::vector<ValueType*>& dataChunks,
						 const size_t chunkLength,
						 const difference_type offset)
		: currentPtr_(ptr)
		, dataChunks_(dataChunks)
		, chunkLength_(chunkLength)
		, offset_(offset)
	{}

	/// Updates the pointer value according to current offset. It is assumed the offset is valid.
	void _updatePointer()
	{
		const auto chunkIndex = (static_cast<uint32_t>(offset_) / chunkLength_);

		currentPtr_ = dataChunks_[chunkIndex] + (static_cast<uint32_t>(offset_) % chunkLength_);
	}

	/// Tells whether the current offset is within the spanned data.
	bool _isOffsetValid() const
	{
		return (offset_ >= 0) && (static_cast<uint32_t>(offset_) < (dataChunks_.size() * chunkLength_));
	}

	pointer currentPtr_;
	const std::vector<ValueType*>& dataChunks_;
	size_t chunkLength_;
	difference_type offset_;
};
} // namespace mlCore

#endif
