#ifndef MLCORE_BASICTENSOR_H
#define MLCORE_BASICTENSOR_H

#include "LoggingLib/LoggingLib.h"
#include "MLCore/Utilities.h"
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
 * @tparam valueType Type of the underlying data
 */
template <typename valueType>
class BasicTensor
{
	friend class TensorOperations;
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
	BasicTensor(const std::vector<size_t>& shape, const valueType initVal);
	/**
	 * @brief Constructs a new tensor from shape and gives it initial values
	 * 
	 * @param shape Tensor's initial shape
	 * @param initValues Values to assign, there is no check of the init list length
	 */
	BasicTensor(const std::vector<size_t>& shape,
				const std::initializer_list<valueType> initValues);
	~BasicTensor();

	// assign
	BasicTensor& operator=(const BasicTensor&); // copy assignment
	BasicTensor& operator=(BasicTensor&&); // move assignment
	/**
	 * @brief fills tensor with passed value
	 * 
	 * @return
	 */
	BasicTensor& operator=(const valueType);

	// getters
	const std::vector<size_t>& shape() const noexcept
	{
		return shape_;
	}
	size_t nDimensions() const noexcept
	{
		return shape_.size();
	}
	size_t size() const noexcept
	{
		return length_;
	}

	Iterator begin() const
	{
		return Iterator(data_);
	}

	Iterator end() const
	{
		return Iterator(data_ + length_);
	}

	// setters
	void reshape(const std::vector<size_t>&);
	/**
	 * @brief assigns new values to tensor in places specified by axes ranges
	 * 
	 * @param indices list of ranges through each axis that will be taken into account while assigning new data
	 * @param newData list of values to assign
	 * @param wrapData whether the values should be repeated to fit. If false and there are to few values, an exception will be raised
	 */
	void assign(std::initializer_list<std::pair<size_t, size_t>> indices,
				std::initializer_list<valueType> newData,
				const bool wrapData = false);

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
		InputIter collectionIter = first;
		for(size_t i = 0; i < length_; i++)
		{

			if(collectionIter >= last)
			{
				if(!wrapData)
					throw std::out_of_range("Too few values to assign to the tensor.");

				collectionIter = first;
			}

			data_[i] = *collectionIter;
			collectionIter++;
		}

		if((collectionIter < last) && (!wrapData))
			throw std::out_of_range("Too many values to assign to the tensor.");
	}

	inline void fill(std::initializer_list<valueType> newData, const bool wrapData = false)
	{
		fill(newData.begin(), newData.end(), wrapData);
	}

	void fill(const ITensorInitializer<valueType>&& initializer);

	// operators
	BasicTensor operator+(const BasicTensor&) const;
	BasicTensor operator-(const BasicTensor&) const;
	BasicTensor operator*(const BasicTensor&) const;
	BasicTensor operator/(const BasicTensor&) const;
	BasicTensor operator-() const;

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

private:
	struct Iterator
	{
		using iterator_category = std::input_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = valueType;
		using pointer = valueType*;
		using reference = valueType&;

		Iterator(pointer ptr)
			: curr_ptr(ptr)
		{ }

		reference operator*()
		{
			return *curr_ptr;
		}
		pointer operator->()
		{
			return curr_ptr;
		}
		Iterator& operator++()
		{
			curr_ptr++;
			return *this;
		}
		Iterator operator++(int)
		{
			auto tmp = *this;
			++(*this);
			return tmp;
		}

		friend bool operator==(const Iterator& iter1, const Iterator& iter2)
		{
			return iter1.curr_ptr == iter2.curr_ptr;
		}

		friend bool operator!=(const Iterator& iter1, const Iterator& iter2)
		{
			return iter1.curr_ptr != iter2.curr_ptr;
		}
		friend bool operator<(const Iterator& iter1, const Iterator& iter2)
		{
			return iter1.curr_ptr < iter2.curr_ptr;
		}

	private:
		pointer curr_ptr;
	};

private:
	void checkIndicesList_(
		const std::initializer_list<std::pair<size_t, size_t>>::const_iterator beg,
		const std::initializer_list<std::pair<size_t, size_t>>::const_iterator end) const;
	void checkShape_(const std::vector<size_t>&) const;

	BasicTensor
	performOperation_(const BasicTensor&,
					  const std::function<valueType(const valueType, const valueType)>&) const;

	static const inline std::function<valueType(const valueType l, const valueType r)>
		plusOperator_ = [](const valueType l, const valueType r) { return l + r; };

	static const inline std::function<valueType(const valueType l, const valueType r)>
		minusOperator_ = [](const valueType l, const valueType r) { return l - r; };

	static const inline std::function<valueType(const valueType l, const valueType r)>
		mulOperator_ = [](const valueType l, const valueType r) { return l * r; };

	static const inline std::function<valueType(const valueType l, const valueType r)>
		divOperator_ = [](const valueType l, const valueType r) { return l / r; };

private:
	size_t length_;
	std::vector<size_t> shape_;
	valueType* data_;
};

template <typename TensorValueType>
std::ostream& operator<<(std::ostream& out, const BasicTensor<TensorValueType>& tensor)
{

	const auto blockSize = std::accumulate(
		tensor.begin(), tensor.end(), size_t(0), [](const auto currMax, const auto& element) {
			return std::max(currMax, (std::ostringstream() << element).str().length());
		});

	out << "<BasicTensor dtype=" << typeid(TensorValueType).name()
		<< " shape=" << stringifyVector(tensor.shape_) << ">";

	std::function<void(typename std::vector<size_t>::const_iterator,
					   const TensorValueType*,
					   const std::string&,
					   size_t)>
		recursePrint;
	recursePrint =
		[&blockSize, &recursePrint, &out, &tensor](std::vector<size_t>::const_iterator shapeIter,
												   const TensorValueType* dataPtr,
												   const std::string& preamble,
												   size_t offset) {
			offset /= *shapeIter;
			if(shapeIter == std::prev(tensor.shape_.end()))
			{
				out << "\n" << preamble << "[";

				size_t i;
				for(i = 0; i < (*shapeIter) - 1; i++)
					out << std::setw(blockSize) << dataPtr[i] << ", ";
				out << std::setw(blockSize) << dataPtr[i];

				out << "]";
			}
			else
			{
				out << "\n" << preamble << "[";

				size_t i;
				for(i = 0; i < (*shapeIter); i++)
				{
					recursePrint(shapeIter + 1, dataPtr + i * offset, preamble + " ", offset);
					// out << ",";
				}
				// recursePrint(shapeIter + 1, dataPtr + i * (*shapeIter), preamble + " ");

				out << "\n" << preamble << "]";
			}
		};

	// for traversing data
	const size_t offset = std::accumulate(
		tensor.shape_.begin(), tensor.shape_.end(), 1, [](const size_t curr, const size_t dim) {
			return curr * dim;
		});

	recursePrint(tensor.shape_.cbegin(), tensor.data_, "", offset);

	return out;
}

using Tensor = BasicTensor<double>;
using TensorPtr = std::shared_ptr<Tensor>;
} // namespace mlCore

#endif