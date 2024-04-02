#ifndef MLCORE_INCLUDE_MLCORE_TENSORITERATOR_HPP
#define MLCORE_INCLUDE_MLCORE_TENSORITERATOR_HPP

#include <iterator>

namespace mlCore
{

template <typename ValueType>
class TensorIterator
{
public:
	// NOLINTBEGIN
	using iterator_category = std::random_access_iterator_tag;
	using value_type = ValueType;
	using difference_type = std::ptrdiff_t;
	using pointer = ValueType*;
	using reference = ValueType&;

	// NOLINTEND

	/**
	 * @brief Constructs a new TensorIterator with initial position.
	 *
	 * @param ptr Initially pointed memory.
	 */
	explicit TensorIterator(pointer ptr)
		: _currPtr(ptr)
	{}

	/**
	 * @brief Dereference operator.
	 *
	 * @return reference
	 */
	reference operator*()
	{
		return *_currPtr;
	}

	/**
	 * @brief Class member access operator.
	 *
	 * @return pointer
	 */
	pointer operator->()
	{
		return _currPtr;
	}

	/**
	 * @brief Prefix increment operator.
	 *
	 * @return Iterator&
	 */
	TensorIterator& operator++()
	{
		_currPtr++;
		return *this;
	}

	// NOLINTBEGIN
	/**
	 * @brief Postfix increment operator.
	 *
	 * @return const Iterator
	 */
	TensorIterator operator++(int)
	{
		auto tmp = *this;
		++(*this);
		return tmp;
	}

	// NOLINTEND

	TensorIterator& operator+=(difference_type n)
	{
		_currPtr += n;
		return *this;
	}

	TensorIterator& operator-=(difference_type n)
	{
		_currPtr -= n;
		return *this;
	}

	difference_type operator-(const TensorIterator& other) const
	{
		return _currPtr - other._currPtr;
	}

	/**
	 * @brief Compares two operators.
	 *
	 * @param other Iterator to compare.
	 * @return true If iterators are equal.
	 * @return false In the opposite case.
	 */
	bool operator==(const TensorIterator& other) const
	{
		return _currPtr == other._currPtr;
	}

	/**
	 * @brief Checks if two tensors differ.
	 *
	 * @param other Iterator to compare.
	 * @return true If the tensors are different.
	 * @return false In the opposite case.
	 */
	bool operator!=(const TensorIterator& other) const
	{
		return _currPtr != other._currPtr;
	}

	/**
	 * @brief Checks if TensorIterator comes first in order before the other one.
	 *
	 * @param other Iterator to compare.
	 * @return true If `this` is smaller than the `other`.
	 * @return false In the opposite case.
	 */
	bool operator<(const TensorIterator& other)
	{
		return _currPtr < other._currPtr;
	}

private:
	pointer _currPtr;
};
} // namespace mlCore

#endif
