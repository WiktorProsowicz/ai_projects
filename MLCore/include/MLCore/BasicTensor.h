#ifndef MLCORE_BASICTENSOR_H
#define MLCORE_BASICTENSOR_H

#include "LoggingLib/LoggingLib.h"
#include "MLCore/Utilities.h"
#include <MLCore/TensorInitializers/ITensorInitializer.h>
#include <algorithm>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <vector>

namespace mlCore
{
/**
 * @brief Class implements a concept of tensor, support basic operation, transposition etc.
 * 
 * @tparam ValueType Type of the underlying data
 */
template <typename ValueType>
class BasicTensor
{
	template <typename OperationsType>
	friend class BasicTensorOperations;

	template <typename OperationsImplType>
	friend class TensorOperationsImpl;

	class Iterator;

public:
	BasicTensor() = delete;
	BasicTensor(const BasicTensor&); // copy constructor
	BasicTensor(BasicTensor&&); // move constructor
	/**
	 * @brief constructs tensor from shape
	 * 
	 */
	BasicTensor(const std::vector<size_t>&);
	/**
	 * @brief constructs tensor from shape and fills it with initial value
	 * 
	 * @param shape tensor's initial shape
	 * @param initVal initital value
	 */
	BasicTensor(const std::vector<size_t>& shape, ValueType initVal);
	/**
	 * @brief Constructs a new tensor from shape and gives it initial values
	 * 
	 * @param shape Tensor's initial shape
	 * @param initValues Values to assign, there is no check of the init list length
	 */
	BasicTensor(const std::vector<size_t>& shape, std::initializer_list<ValueType> initValues);
	~BasicTensor();

	// assign
	BasicTensor& operator=(const BasicTensor&); // copy assignment
	BasicTensor& operator=(BasicTensor&&); // move assignment
	/**
	 * @brief fills tensor with passed value
	 * 
	 * @return
	 */
	BasicTensor& operator=(ValueType);

	/// Gets tensor's shape
	const std::vector<size_t>& shape() const noexcept
	{
		return shape_;
	}

	/// Gets number of tensor's dimensions
	size_t nDimensions() const noexcept
	{
		return shape_.size();
	}

	/// Gets number of tensor's elements
	size_t size() const noexcept
	{
		return length_;
	}

	/// Gets beginning tensor's iterator
	Iterator begin() const
	{
		return Iterator(data_);
	}

	/// Gets ending tensor's iterator
	Iterator end() const
	{
		return Iterator(data_ + length_);
	}

	/**
	 * @brief Changes shape of the tensor. Basic checks over the given shape are performed
	 * 
	 * @param newShape The new shape to assign
	 */
	void reshape(const std::vector<size_t>& newShape);

	/**
	 * @brief assigns new values to tensor in places specified by axes ranges
	 * 
	 * @param indices list of ranges through each axis that will be taken into account while assigning new data
	 * @param newData list of values to assign
	 * @param wrapData whether the values should be repeated to fit. If false and there are to few values, an exception will be raised
	 */
	void assign(std::initializer_list<std::pair<size_t, size_t>> indices,
				std::initializer_list<ValueType> newData,
				bool wrapData = false);

	BasicTensor operator+(const BasicTensor& other) const;
	BasicTensor operator-(const BasicTensor& other) const;
	BasicTensor operator*(const BasicTensor& other) const;
	BasicTensor operator/(const BasicTensor& other) const;
	BasicTensor operator-() const;

	BasicTensor& operator+=(const BasicTensor& other);
	BasicTensor& operator-=(const BasicTensor& other);
	BasicTensor& operator*=(const BasicTensor& other);
	BasicTensor& operator/=(const BasicTensor& other);

	BasicTensor matmul(const BasicTensor&) const;

	/**
	 * @brief Create copy of the tensor and return its transposed version
	 * 
	 * @return Transposed tensor
	 */
	BasicTensor transposed() const;

	// displaying
	template <typename TensorValueType>
	friend std::ostream& operator<<(std::ostream&, const BasicTensor<TensorValueType>&);

	/**
	 * @brief assigns values to the tensor
	 * 
	 * @tparam InputIter type of iterator to take data from
	 * @param first beginning iterator of values collection
	 * @param last ending iterator
	 * @param wrapData whether the values should be repeated to fit. If false and there are to few values, an exception will be raised
	 */
	template <typename InputIter>
	void fill(InputIter first, InputIter last, const bool wrapData = false)
	{
		const auto nElementsToAssign = static_cast<size_t>(std::distance(first, last));

		if(!wrapData)
		{
			if(size() < nElementsToAssign)
			{
				throw std::out_of_range("Too many values to assign to the tensor.");
			}

			if(size() > nElementsToAssign)
			{
				throw std::out_of_range("Too few values to assign to the tensor.");
			}
		}

		InputIter collectionIter = first;
		for(size_t i = 0; i < length_; i++)
		{

			if(collectionIter >= last)
			{
				collectionIter = first;
			}

			data_[i] = *collectionIter;
			collectionIter++;
		}
	}

	inline void fill(std::initializer_list<ValueType> newData, const bool wrapData = false)
	{
		fill(newData.begin(), newData.end(), wrapData);
	}

	void fill(const tensorInitializers::ITensorInitializer<ValueType>& initializer);

private:
	class Iterator
	{
	public:
		using iterator_category = std::input_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = ValueType;
		using pointer = ValueType*;
		using reference = ValueType&;

		Iterator(pointer ptr)
			: currPtr_(ptr)
		{ }

		reference operator*()
		{
			return *currPtr_;
		}
		pointer operator->()
		{
			return currPtr_;
		}
		Iterator& operator++()
		{
			currPtr_++;
			return *this;
		}

		// NOLINTBEGIN(readability-const-return-type)
		const Iterator operator++(int)
		{
			auto tmp = *this;
			++(*this);
			return tmp;
		}
		// NOLINTEND(readability-const-return-type)

		friend bool operator==(const Iterator& iter1, const Iterator& iter2)
		{
			return iter1.currPtr_ == iter2.currPtr_;
		}

		friend bool operator!=(const Iterator& iter1, const Iterator& iter2)
		{
			return iter1.currPtr_ != iter2.currPtr_;
		}
		friend bool operator<(const Iterator& iter1, const Iterator& iter2)
		{
			return iter1.currPtr_ < iter2.currPtr_;
		}

	private:
		pointer currPtr_;
	};

private:
	void
	checkIndicesList_(std::initializer_list<std::pair<size_t, size_t>>::const_iterator beg,
					  std::initializer_list<std::pair<size_t, size_t>>::const_iterator end) const;
	void checkShape_(const std::vector<size_t>&) const;

private:
	size_t length_;
	std::vector<size_t> shape_;
	ValueType* data_;
};

template <typename TensorValueType>
std::ostream& operator<<(std::ostream& out, const BasicTensor<TensorValueType>& tensor);

using Tensor = BasicTensor<double>;
using TensorPtr = std::shared_ptr<Tensor>;
} // namespace mlCore

#endif