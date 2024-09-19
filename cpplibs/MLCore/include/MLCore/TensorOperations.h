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

	/// @brief Performs matrix transposition on `arg`.
	static BasicTensor<ValueType> transpose(const BasicTensor<ValueType>& arg);

	/// @brief Performs matrix multiplication between `lhs` and `rhs`.
	/// @param extendLhs Tells whether the lhs' shape should be right-extended with 1.
	/// @param extentRhs Tells whether the rhs' shape should be right-extended with 1.
	static BasicTensor<ValueType> matmul(const BasicTensor<ValueType>& lhs,
										 const BasicTensor<ValueType>& rhs,
										 const bool extendLhs,
										 const bool extendRhs);

	/**
	 * @brief Creates tensor from compile-time nested initializer list form.
	 *
	 * @param tensorForm Literal-like tensor values having the desired tensor's shape.
	 * @return Tensor created from the given `tensorForm`.
	 */
	static BasicTensor<ValueType> makeTensor(const TensorForm<ValueType>& tensorForm);
};

using TensorOperations = BasicTensorOperations<double>;
} // namespace mlCore

#endif
