#include "MLCore/Utilities.h"

#include <vector>

#include "MLCore/UtilitiesImpl.h"

namespace mlCore::detail
{
std::vector<size_t> getOutputShapeForMatmul(const std::vector<size_t>& lhsShape,
											const std::vector<size_t>& rhsShape)
{
	assertCanMatmulTensors(lhsShape, rhsShape);

	const auto [lhsPaddedShape, rhsPaddedShape] = padShapes(lhsShape, rhsShape);

	return getReturnShapeForMatmul(lhsPaddedShape, rhsPaddedShape);
}
} // namespace mlCore::detail
