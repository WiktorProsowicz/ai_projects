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
 * @brief Creates a human-readable serialized form of the vector. Can be used for displaying tensors' shapes
 * etc.
 *
 * @param vect Vector to be serialized.
 * @param openSign Character serving as the beginning for the result sequence.
 * @param closeSign Character serving as the end for the result sequence.
 * @return Serialized representation of the vector.
 *
 * @example
 *
 * stringifyVector(std::vector<uint32_t>{0, 1, 2, 3}, '(', ')') -> (0, 1, 2, 3)
 */
template <typename T>
std::string stringifyVector(const std::vector<T>& vector,
							const char* const openSign = "(",
							const char* const closeSign = ")")
{
	return fmt::format("{}{}{}", openSign, fmt::join(vector, ", "), closeSign);
}

template <typename BaseType>
using TensorForm = detail::RawTensorForm<BaseType>;

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
