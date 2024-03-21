#ifndef MLCORE_INCLUDE_MLCORE_TENSORITERATOR_HPP
#define MLCORE_INCLUDE_MLCORE_TENSORITERATOR_HPP

#include <iterator>

namespace mlCore
{

template <typename ValueType>
class TensorIterator
{
public:
	using iterator_category = std::random_access_iterator_tag;
	using value_type = ValueType;
	using difference_type = std::ptrdiff_t;
	using pointer = ValueType*;
	using reference = ValueType&;

	/**
     * @brief Constructs a new TensorIterator with initial position.
     * 
     * @param ptr Initially pointed memory.
     */
	explicit TensorIterator(pointer ptr)
		: currPtr_(ptr)
	{ }

	/**
     * @brief Dereference operator.
     * 
     * @return reference
     */
	reference operator*()
	{
		return *currPtr_;
	}

	/**
     * @brief Class member access operator.
     * 
     * @return pointer 
     */
	pointer operator->()
	{
		return currPtr_;
	}

	/**
     * @brief Prefix increment operator.
     * 
     * @return Iterator& 
     */
	TensorIterator& operator++()
	{
		currPtr_++;
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
		currPtr_ += n;
		return *this;
	}

	TensorIterator& operator-=(difference_type n)
	{
		currPtr_ -= n;
		return *this;
	}

	difference_type operator-(const TensorIterator& other) const
	{
		return currPtr_ - other.currPtr_;
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
		return currPtr_ == other.currPtr_;
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
		return currPtr_ != other.currPtr_;
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
		return currPtr_ < other.currPtr_;
	}

private:
	pointer currPtr_;
};
} // namespace mlCore

#endif