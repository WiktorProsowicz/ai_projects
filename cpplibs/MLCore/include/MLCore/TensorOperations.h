#ifndef MLCORE_INCLUDE_MLCORE_TENSOROPERATIONS_H
#define MLCORE_INCLUDE_MLCORE_TENSOROPERATIONS_H

#include <cmath>

#include "MLCore/BasicTensor.h"
#include "MLCore/Utilities.h"

namespace mlCore
{
/**
 * @brief Set of both binary and unary operators for tensors
 *
 */
template <typename ValueType>
class BasicTensorOperations
{
public:
	/// @brief Computes result of the `lhs` to the power of `rhs`.
	static BasicTensor<ValueType> power(const BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs);

	/// @brief Computes natural logarithm of the `arg`.
	static BasicTensor<ValueType> ln(const BasicTensor<ValueType>& arg);

	/// @brief Computes REctified Linear Unit result of `arg`.
	static BasicTensor<ValueType> relu(const BasicTensor<ValueType>& arg);

	/// @brief Computes sigmoid function result of `arg`.
	static BasicTensor<ValueType> sigmoid(const BasicTensor<ValueType>& arg);

	/**
	 * @brief Performs matrix transposition on `arg`.
	 */
	static BasicTensor<ValueType> transpose(const BasicTensor<ValueType>& arg,
											MatrixSpec spec = MatrixSpec::Default);

	/**
	 * @brief Performs matrix multiplication between `lhs` and `rhs`.
	 */
	static BasicTensor<ValueType> matmul(const BasicTensor<ValueType>& lhs,
										 const BasicTensor<ValueType>& rhs,
										 MatrixSpec lhsSpec = MatrixSpec::Default,
										 MatrixSpec rhsSpec = MatrixSpec::Default);

	/**
	 * @brief Creates tensor from compile-time nested initializer list form.
	 *
	 * @param tensorForm Literal-like tensor values having the desired tensor's shape.
	 * @return Tensor created from the given `tensorForm`.
	 */
	static BasicTensor<ValueType> makeTensor(const TensorForm<ValueType>& tensorForm);

	/**
	 * @brief Reduces a given tensor to a target shape by summing up the elements.
	 *
	 * @param arg Tensor to be reduced.
	 * @param targetShape Shape to which the `arg` should be reduced. The shape should be right-aligned with
	 * the original shape, i.e. the original shape should be its left-hand extension.
	 */
	static BasicTensor<ValueType> reduceAdd(const BasicTensor<ValueType>& arg,
											const std::vector<size_t>& targetShape);
};

using TensorOperations = BasicTensorOperations<double>;
} // namespace mlCore

#endif
