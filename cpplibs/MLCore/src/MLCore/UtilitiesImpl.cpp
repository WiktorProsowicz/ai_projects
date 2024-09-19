#include "MLCore/UtilitiesImpl.h"

#include <LoggingLib/LoggingLib.hpp>
#include <fmt/format.h>

namespace mlCore::detail
{
void assertCanMatmulTensors(const std::vector<size_t>& lhsShape, const std::vector<size_t>& rhsShape)
{
	// For clean error throwing with additional info about shapes
	const auto throwInformative = [&lhsShape, &rhsShape](const std::string& message)
	{
		LOG_ERROR("MLCore::TensorOperations",
				  fmt::format("Cannot perform matrix multiplication for shapes '{}' and '{}' - {}.",
							  stringifyVector(lhsShape),
							  stringifyVector(rhsShape),
							  message));
	};

	if((lhsShape.size() < 2) || (rhsShape.size() < 2))
	{
		throwInformative("Tensors have to have at least 2 dimensions");
	}

	if(lhsShape[lhsShape.size() - 1] != rhsShape[rhsShape.size() - 2])
	{
		throwInformative("Tensors' shapes are incompatible");
	}

	const auto [lhsPaddedShape, rhsPaddedShape] = padShapes(lhsShape, rhsShape);

	for(size_t i = 0; i < lhsPaddedShape.size() - 2; i++)
	{
		if((lhsPaddedShape[i] != rhsPaddedShape[i]) && (lhsPaddedShape[i] != 1) && (rhsPaddedShape[i] != 1))
		{
			throwInformative("Cannot broadcast tensors for the operation");
		}
	}
}

std::pair<std::vector<size_t>, std::vector<size_t>> padShapes(const std::vector<size_t>& shape1,
															  const std::vector<size_t>& shape2)
{
	const size_t biggerSize = std::max(shape1.size(), shape2.size());

	std::vector<size_t> paddedShape1(biggerSize, 1);
	std::vector<size_t> paddedShape2(biggerSize, 1);

	std::copy(shape1.cbegin(),
			  shape1.cend(),
			  std::next(paddedShape1.begin(), static_cast<ptrdiff_t>(biggerSize - shape1.size())));

	std::copy(shape2.cbegin(),
			  shape2.cend(),
			  std::next(paddedShape2.begin(), static_cast<ptrdiff_t>(biggerSize - shape2.size())));

	return {paddedShape1, paddedShape2};
}

std::vector<size_t> getReturnShapeForMatmul(const std::vector<size_t>& lhsPaddedShape,
											const std::vector<size_t>& rhsPaddedShape)
{
	std::vector<size_t> retShape(lhsPaddedShape.size());
	const auto& shapeSize = lhsPaddedShape.size();

	retShape[shapeSize - 2] = lhsPaddedShape[shapeSize - 2];
	retShape[shapeSize - 1] = rhsPaddedShape[shapeSize - 1];

	for(size_t i = 0; i < shapeSize - 2; i++)
	{
		retShape[i] = lhsPaddedShape[i] == 1 ? rhsPaddedShape[i] : lhsPaddedShape[i];
	}

	return retShape;
}
} // namespace mlCore::detail