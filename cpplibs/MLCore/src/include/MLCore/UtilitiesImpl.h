#ifndef MLCORE_SRC_INCLUDE_MLCORE_UTILITIESIMPL_H
#define MLCORE_SRC_INCLUDE_MLCORE_UTILITIESIMPL_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "MLCore/BasicTensor.h"

namespace mlCore
{
enum class MatrixSpec : std::uint8_t;
} // namespace mlCore

namespace mlCore::detail
{
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

/// @brief Extends a given shape according to provided matrix specification.
std::vector<size_t> applyMatSpecToShape(const std::vector<size_t>& shape, MatrixSpec spec);

/// @brief Checks if the input shape is a row or a column vector, i.e. has has the shape in the form (..., 1,
/// n) or (..., n, 1).
bool isRowOrColumnVector(const std::vector<size_t>& shape);

/// @brief If the input shape is a row or a column vector, trims the '1' dimension.
std::vector<size_t> trimRowOrColumnVector(const std::vector<size_t>& shape);

/**
 * @brief Checks if two tensors have correct shapes to be matrix-multiplied.
 * @param lhsShape Shape of the lhs tensor.
 * @param rhsShape Shape of the rhs tensor.
 */
void assertCanMatmulTensors(const std::vector<size_t>& lhsShape, const std::vector<size_t>& rhsShape);

/// @brief Pads two shapes with 1s to have the same number of dimensions.
/// TODO: Move to .cpp file and find out why IWYU crashes if the function is not inline.
inline std::pair<mlCore::TensorShape, mlCore::TensorShape> padShapes(const std::vector<size_t>& shape1,
																	 const std::vector<size_t>& shape2)
{
	const auto biggerSize = std::max(shape1.size(), shape2.size());

	std::vector<size_t> paddedShape1(biggerSize, 1);
	std::vector<size_t> paddedShape2(biggerSize, 1);

	std::ranges::copy(shape1.cbegin(),
					  shape1.cend(),
					  std::next(paddedShape1.begin(), static_cast<ptrdiff_t>(biggerSize - shape1.size())));

	std::ranges::copy(shape2.cbegin(),
					  shape2.cend(),
					  std::next(paddedShape2.begin(), static_cast<ptrdiff_t>(biggerSize - shape2.size())));

	return {paddedShape1, paddedShape2};
}

/// @brief Computes the shape of the result of a matrix multiplication.
std::vector<size_t> getReturnShapeForMatmul(const std::vector<size_t>& lhsPaddedShape,
											const std::vector<size_t>& rhsPaddedShape);

/// @brief Tells if the given `shape` is the right part of the `targetShape`.
bool isShapeExtendableToAnother(const std::vector<size_t>& shape, const std::vector<size_t>& targetShape);

} // namespace mlCore::detail

#endif
