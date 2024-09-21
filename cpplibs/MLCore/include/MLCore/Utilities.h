#ifndef MLCORE_UTILITIES_H
#define MLCORE_UTILITIES_H

#include <iterator>
#include <limits>
#include <sstream>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace mlCore
{

namespace detail
{
template <typename ContainedType>
class RawTensorInitList;

/**
 * @brief Variant used for making compile-time "tensor literals".
 *
 * Usage of the RawTensorForm mimics the numpy.array function i.e tensor = np.array([[1, 2, 3], [4, 5, 6]]).
 *
 * @tparam BaseType Indivisible underlying type of the desired tensor.
 *
 */
template <typename BaseType>
using RawTensorForm = std::variant<BaseType, detail::RawTensorInitList<BaseType>>;

/**
 * @brief Container-type tensor form for nested objects.
 *
 * @tparam ContainedType Underlying object's type held by the initializer list.
 * Can be either the destination tensor's data type or another init list.
 */
template <typename ContainedType>
class RawTensorInitList : public std::vector<RawTensorForm<ContainedType>>
{
public:
	using std::vector<RawTensorForm<ContainedType>>::vector;
};

/**
 * @brief Computes the shape of the result of a matrix multiplication.
 *
 * @param lhsShape Shape of the left-hand side matrix.
 * @param rhsShape Shape of the right-hand side matrix.
 *
 * @throws std::runtime_error If the shapes are incompatible.
 */
std::vector<size_t> getOutputShapeForMatmul(const std::vector<size_t>& lhsShape,
											const std::vector<size_t>& rhsShape);
} // namespace detail

/**
 * @brief Represents an additional specification telling how a tensor should be treated by an algorithm.
 *
 * @details Whenever it is required to treat the last dimension of a tensor as either a row or a column
 * vector, which is in practice represented by two-dimensional matrix, an additional value may specify whether
 * its shape should be treated in a different way without the need of explicitly reshaping the tensor.
 *
 * @example
 *
 * // Implicitly treated as a (10, 1) coumn vector.
 * const mlCore::Tensor tensor(mlCore::TensorShape{10});
 *
 * // Although the tensor is only one-dimensional, a (1, 10) matrix is created.
 * const auto transposed = mlCore::TensorOperations::transpose(tensor, mlCore::MatrixSpec::ColumnVector);
 *
 * // The same applies for tensors having more than one dimension.
 * // e.g. a (batch_size, 10) tensor may be treated as a batch_size * (1, 10) row vectors.
 */
enum class MatrixSpec
{
	/// Treat the last dimension as a column vector. (As if there was a '1' at the end of the
	/// shape.)
	ColumnVector,
	/// Treat the last dimension as a row vector. (As if there was a '1' before the last
	/// dimension.)
	RowVector,
	/// Leave the tensor as it is.
	Default
};

/// @brief Represents an element or a slice of a tensor that is being created e.g. via a tensor literal.
/// @see TensorOperations::makeTensor
template <typename BaseType>
using TensorForm = detail::RawTensorForm<BaseType>;

/// @brief Represents a slice of tensor that is being created e.g. via a tensor literal.
/// @see TensorOperations::makeTensor
template <typename BaseType>
using TensorArr = detail::RawTensorInitList<BaseType>;

/**
 * @brief Represents a part of the tensor spanned by a tensor slice.
 *
 * Each pair of indices indicates a part of a specific dimension in tensor's shape.
 *
 */
using SliceIndices = std::vector<std::pair<size_t, size_t>>;

} // namespace mlCore

#endif
