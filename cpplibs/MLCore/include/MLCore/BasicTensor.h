#ifndef MLCORE_BASICTENSOR_H
#define MLCORE_BASICTENSOR_H

// __C++ standard headers__
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

// __Own software headers__
#include <LoggingLib/LoggingLib.hpp>
#include <MLCore/Utilities.h>
#include <MLCore/TensorInitializers/ITensorInitializer.hpp>
#include <MLCore/TensorIterator.hpp>
#include <MLCore/BasicTensorSlice.h>

namespace mlCore
{
namespace detail
{
template <typename ValueType>
class TensorOperationsImpl;
}

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
	friend class detail::TensorOperationsImpl;

	template <typename BasicTensorSliceType>
	friend class BasicTensorSlice;

public:
	/**
	 * @brief Constructs a new scalar-type tensor.
	 * 
	 */
	BasicTensor();

	/**
	 * @brief Constructs a new scalar-type tensor with initial value. Useful as a conversion from ValueType.
	 * Example:
	 * 
	 * tensor : BasicTensor<double>
	 * 
	 * tensor + 5.0   ->   tensor + BasicTensor<double>(5.0)
	 * 
	 * @param initVal 
	 */
	BasicTensor(ValueType initVal);

	/**
	 * @brief Copy constructor.
	 * 
	 * @param other Tensor to copy.
	 */
	BasicTensor(const BasicTensor& other);

	/**
	 * @brief Move constructor.
	 * 
	 * @param other Tensor to move.
	 */
	BasicTensor(BasicTensor&& other);

	/**
	 * @brief Construct a new tensor with given shape.
	 * 
	 * @param shape Tensor's initial shape.
	 */
	BasicTensor(const std::vector<size_t>& shape);

	/**
	 * @brief Constructs tensor from shape and fills it with initial value.
	 * 
	 * @param shape Tensor's initial shape.
	 * @param initVal Initial value for the whole tensor's data.
	 */
	BasicTensor(const std::vector<size_t>& shape, ValueType initVal);

	/**
	 * @brief Constructs a new tensor from shape and gives it initial values.
	 * 
	 * @param shape Tensor's initial shape.
	 * @param initValues Values to assign, there is no check of the init list length.
	 */
	BasicTensor(const std::vector<size_t>& shape, std::initializer_list<ValueType> initValues);

	/**
	 * @brief Tensor's destructor releasing the resources.
	 * 
	 */
	~BasicTensor();

	/**
	 * @brief Copy assignment operator.
	 * 
	 * @param other Tensor to copy.
	 * @return BasicTensor& 
	 */
	BasicTensor& operator=(const BasicTensor& other);

	/**
	 * @brief Move assignment operator.
	 * 
	 * @param other Tensor to move.
	 * @return BasicTensor& 
	 */
	BasicTensor& operator=(BasicTensor&& other);

	/// Gets tensor's shape.
	const std::vector<size_t>& shape() const noexcept
	{
		return shape_;
	}

	/// Gets number of tensor's dimensions.
	size_t nDimensions() const noexcept
	{
		return shape_.size();
	}

	/// Gets number of tensor's elements.
	size_t size() const noexcept
	{
		return length_;
	}

	/// Gets beginning tensor's iterator.
	inline TensorIterator<ValueType> begin() const
	{
		return TensorIterator<ValueType>(data_);
	}

	/// Gets ending tensor's iterator.
	inline TensorIterator<ValueType> end() const
	{
		return TensorIterator<ValueType>(data_ + length_);
	}

	/**
	 * @brief Changes shape of the tensor. Basic checks over the given shape are performed.
	 * 
	 * @param newShape The new shape to assign.
	 */
	void reshape(const std::vector<size_t>& newShape);

	/**
	 * @brief Assigns new values to tensor in places specified by axes ranges.
	 * 
	 * @param indices List of ranges through each axis that will be taken into account while assigning new data.
	 * @param newData List of values to assign.
	 * @param wrapData Whether the values should be repeated to fit. If false and there are to few values, an exception will be raised.
	 */
	void assign(std::initializer_list<std::pair<size_t, size_t>> indices,
				std::initializer_list<ValueType> newData,
				bool wrapData = false);

	/// Creates a product of adding `this` with `other` tensor.
	BasicTensor operator+(const BasicTensor& other) const;

	/// Creates a product of subtracting `other` tensor from `this`.
	BasicTensor operator-(const BasicTensor& other) const;

	/// Creates a product of multiplying `this` by `other` tensor.
	BasicTensor operator*(const BasicTensor& other) const;

	/// Creates a product of dividing `this` by `other` tensor.
	BasicTensor operator/(const BasicTensor& other) const;

	/// Adds `other` tensor to `this`.
	BasicTensor& operator+=(const BasicTensor& other);

	/// Subtract `other` tensor from `this`.
	BasicTensor& operator-=(const BasicTensor& other);

	/// Multiplies `this` by `other` tensor.
	BasicTensor& operator*=(const BasicTensor& other);

	/// Divides `this` by `other` tensor.
	BasicTensor& operator/=(const BasicTensor& other);

	/// Creates the negation of `this`.
	BasicTensor operator-() const;

	/**
	 * @brief Performs matrix multiplication operation on `this` and `other` tensor.
	 * 
	 * @param other Tensor to matrix-multiply `this` by.
	 * @return BasicTensor Product of matrix multiplication.
	 */
	BasicTensor matmul(const BasicTensor& other) const;

	/**
	 * @brief Creates transposed version of `this`.
	 * 
	 * @return Transposed tensor.
	 */
	BasicTensor transposed() const;

	/**
	 * @brief Fills the tensor with given data.
	 * 
	 * @param newData Data to assign to tensor.
	 * @param wrapData Whether to repeat the data to fit.
	 */
	void fill(std::initializer_list<ValueType> newData, bool wrapData = false);

	/**
	 * @brief Fills the tensor with data given by initializer.
	 * 
	 * @param initializer Initializer object from which the data to assign is taken.
	 */
	void fill(const tensorInitializers::ITensorInitializer<ValueType>& initializer);

	/**
	 * @brief Assigns values to the tensor.
	 * 
	 * @tparam InputIter Type of iterator to take data from.
	 * @param first Beginning iterator of values collection.
	 * @param last Ending iterator.
	 * @param wrapData Whether the values should be repeated to fit. If false and there are to few values, an exception will be raised.
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

	/**
	 * @brief Creates a view over the tensor's data. The spanned data is determined by the provided indices.
	 * 
	 * @param indices Set of ranges through each axis that will be taken into account while creating the slice.
	 */
	BasicTensorSlice<ValueType> slice(const std::vector<std::pair<size_t, size_t>>& indices);

	template <typename TensorValueType>
	friend std::ostream& operator<<(std::ostream& out, const BasicTensor<TensorValueType>& tensor);

private:
	/// Traverses list of indices and checks ranges correctness. Correct indices specify tensor slice that can be modified via value assignment.
	/// Throws std::out_of_range if upper[i] > shape[i] or 0 > indices.size() > shape_.size().
	/// Indices is a list of pairs of min-max indices from axis zero i.e for tensor([[1, 2], [3, 4]]) -> list{{0, 1}} -> [1, 2].
	template <typename IndicesIter>
	void _checkIndicesList(IndicesIter beg, IndicesIter end) const;

	/// Checks if all of the `shape`'s elements are positive i.e. eligible to be present in the shape.
	static void _checkShapeElementsPositive(const std::vector<size_t>& shape);

	/// Checks if the number of elements produced by `shape` fits in datatype bounds.
	static void _checkShapeFitsInBounds(const std::vector<size_t>& shape);

	/// Checks if the given `shape` is compatible with the number of elements held by the tensor
	void _checkShapeCompatible(const std::vector<size_t>& shape) const;

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