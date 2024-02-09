#ifndef MLCORE_INCLUDE_MLCORE_BASICTENSORSLICE_H
#define MLCORE_INCLUDE_MLCORE_BASICTENSORSLICE_H

// __C++ standard headers__
#include <vector>
#include <cstdio>
#include <functional>

namespace mlCore
{
template <typename ValueType>
class BasicTensor;

/**
 * @brief Represents part of the tensor taken by providing ranges of indices.
 * 
 * Tensor slices are merely the views over mlCore::BasicTensor instances. There can be multiple
 * instances of slices created by providing different sets of indices as long as the referenced
 * tensor is alive, otherwise the slices are not valid. This can be compared to the dangling references problem. 
 * No lifetime tracking is performed automatically.
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
     * @brief Constructs tensor slice copying its configuration and linking it to the tensor associated do `other`.
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

	/**
	 * @brief Assigns `value` to all of the elements spanned by the slice.
	 * 
	 * @param value Value to assign.
	 */
	void assign(ValueType value);

	/// @see BasicTensorSlice::assign(ValueType)
	void assignAdd(ValueType value);

	/// @see BasicTensorSlice::assign(ValueType)
	void assignSubtract(ValueType value);

	/// @see BasicTensorSlice::assign(ValueType)
	void assignDivide(ValueType value);

	/// @see BasicTensorSlice::assign(ValueType)
	void assignMultiply(ValueType value);

	/**
	 * @brief Assigns given `values` to the elements spanned by the slice.
	 * 
	 * @param values Values to assign. Number of elements spanned by the slice should be divisible by the size of `values`.
	 */
	void assign(std::initializer_list<ValueType> values);

	/// @see BasicTensorSlice::assign(std::initializer_list<ValueType>)
	void assignAdd(std::initializer_list<ValueType> values);

	/// @see BasicTensorSlice::assign(std::initializer_list<ValueType>)
	void assignSubtract(std::initializer_list<ValueType> values);

	/// @see BasicTensorSlice::assign(std::initializer_list<ValueType>)
	void assignDivide(std::initializer_list<ValueType> values);

	/// @see BasicTensorSlice::assign(std::initializer_list<ValueType>)
	void assignMultiply(std::initializer_list<ValueType> values);

	/**
	 * @brief Assigns elements between `begin` and `end` to the spanned elements.
	 * 
	 * Number of elements spanned by the slice should be divisible by the distance
	 * between `begin` and `end`, so that the assigned item set can be wrapped.
	 * 
	 * @param begin Begin iterator of the elements to assign.
	 * @param end End iterator of the elements to assign.
	 */
	template <typename InputIter>
	void assign(InputIter begin, InputIter end) requires std::is_same_v<ValueType, std::iter_value_t<InputIter>>
	{
		_assignContiguousData(std::vector<ValueType>(begin, end));
	}

	/// @see BasicTensorSlice::assign(InputIter, InputIter)
	template <typename InputIter>
	void assignAdd(InputIter begin, InputIter end) requires std::is_same_v<ValueType, std::iter_value_t<InputIter>>
	{
		_assignAddContiguousData(std::vector<ValueType>(begin, end));
	}

	/// @see BasicTensorSlice::assign(InputIter, InputIter)
	template <typename InputIter>
	void assignSubtract(InputIter begin, InputIter end) requires std::is_same_v<ValueType, std::iter_value_t<InputIter>>
	{
		_assignSubtractContiguousData(std::vector<ValueType>(begin, end));
	}

	/// @see BasicTensorSlice::assign(InputIter, InputIter)
	template <typename InputIter>
	void assignDivide(InputIter begin, InputIter end) requires std::is_same_v<ValueType, std::iter_value_t<InputIter>>
	{
		_assignDivideContiguousData(std::vector<ValueType>(begin, end));
	}

	/// @see BasicTensorSlice::assign(InputIter, InputIter)
	template <typename InputIter>
	void assignMultiply(InputIter begin, InputIter end) requires std::is_same_v<ValueType, std::iter_value_t<InputIter>>
	{
		_assignMultiplyContiguousData(std::vector<ValueType>(begin, end));
	}

	/**
	 * @brief Performs matrix multiplication between elements spanned by this and the `other`. The result is assigned to tensor associated to `this`.
	 * 
	 * Spanned elements have to conform to the rules applying to the shapes of multiplied matrix.
	 * Additionally, if the elements of the `other` need to be broadcasted, the shapes should be also
	 * compatible. Last two dimensions of the matmul result's shape should be the same as the last two dimensions of the `this`'s span shape.
	 * 
	 * @param other Tensor slice to multiply the spanned elements by.
	 */
	void assignMatmul(const BasicTensorSlice& other);

private:
	/**
	 * @brief Constructs the slice linking it to the tensor and providing referenced span with indices.
	 * 
	 * @param associatedTensor Tensor to be referenced.
	 * @param indices Span indicating the shape of the slice. The length of the indices should be the same
     * as the length of the source tensor's shape.
	 */
	BasicTensorSlice(BasicTensor<ValueType>& associatedTensor, const std::vector<std::pair<size_t, size_t>>& indices);

private:
	/**
	 * @brief Assigns all of the elements from `data` to the referenced span of the connected tensor.
	 * 
	 * @param data Vector holding contiguous data to assign. 
	 */
	void _assignContiguousData(const std::vector<ValueType>& data);

	/// @see BasicTensor::_assignContiguousData(const std::vector<ValueType>)
	void _assignAddContiguousData(const std::vector<ValueType>& data);

	/// @see BasicTensor::_assignContiguousData(const std::vector<ValueType>)
	void _assignSubtractContiguousData(const std::vector<ValueType>& data);

	/// @see BasicTensor::_assignContiguousData(const std::vector<ValueType>)
	void _assignDivideContiguousData(const std::vector<ValueType>& data);

	/// @see BasicTensor::_assignContiguousData(const std::vector<ValueType>)
	void _assignMultiplyContiguousData(const std::vector<ValueType>& data);

	/// Computes the slice's shape based on its indices and the shape of the associated tensor.
	std::vector<size_t> _computeSliceShape() const;

	/// Computes an array of pointers to the places in memory where there are chunks of contiguous memory.
	std::vector<ValueType*> _computeDataPointers() const;

	/// Computes size of the chunk of contiguous memory spanned by the slice.
	size_t _computeChunkLength() const;

private:
	std::reference_wrapper<mlCore::BasicTensor<ValueType>> tensor_;
	std::vector<std::pair<size_t, size_t>> indices_;
};

using TensorSlice = BasicTensorSlice<double>;

} // namespace mlCore

#endif