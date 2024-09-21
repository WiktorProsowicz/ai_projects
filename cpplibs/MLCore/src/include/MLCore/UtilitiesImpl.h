#ifndef MLCORE_SRC_INCLUDE_MLCORE_UTILITIESIMPL_H
#define MLCORE_SRC_INCLUDE_MLCORE_UTILITIESIMPL_H

#include <string>
#include <vector>

#include <fmt/format.h>

#include "MLCore/Utilities.h"

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
std::vector<size_t> applyMatSpecToShape(const std::vector<size_t>& shape, const MatrixSpec spec);

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
std::pair<std::vector<size_t>, std::vector<size_t>> padShapes(const std::vector<size_t>& shape1,
															  const std::vector<size_t>& shape2);

/// @brief Computes the shape of the result of a matrix multiplication.
std::vector<size_t> getReturnShapeForMatmul(const std::vector<size_t>& lhsPaddedShape,
											const std::vector<size_t>& rhsPaddedShape);

} // namespace mlCore::detail

#endif