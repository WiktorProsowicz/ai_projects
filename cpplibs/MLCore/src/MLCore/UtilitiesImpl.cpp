#include "MLCore/UtilitiesImpl.h"

#include <LoggingLib/LoggingLib.hpp>
#include <fmt/format.h>

namespace mlCore::detail
{
std::vector<size_t> applyMatSpecToShape(const std::vector<size_t>& shape, const MatrixSpec spec)
{
	if(shape.size() < 1)
	{
		LOG_ERROR("MLCore", "Cannot apply matrix specification to a shape with less than 1 dimension.");
	}

	auto newShape = shape;

	switch(spec)
	{
	case MatrixSpec::ColumnVector:
		newShape.emplace_back(1);
		break;

	case MatrixSpec::RowVector:
		newShape.insert(std::prev(newShape.cend()), 1);
		break;

	case MatrixSpec::Default:
		break;
	}

	return newShape;
}

bool isRowOrColumnVector(const std::vector<size_t>& shape)
{
	if(shape.size() < 2)
	{
		LOG_ERROR(
			"MLCore",
			"Cannot check if a shape is a row or a column vector for a shape with less than 2 dimensions!");
	}

	return (shape[shape.size() - 1] == 1) || (shape[shape.size() - 2] == 1);
}

std::vector<size_t> trimRowOrColumnVector(const std::vector<size_t>& shape)
{
	if(shape.size() < 2)
	{
		LOG_ERROR("MLCore", "Cannot trim row or column vector for a shape with less than 2 dimensions!");
	}

	auto newShape = shape;

	if(newShape[newShape.size() - 1] == 1)
	{
		newShape.pop_back();
	}
	else if(newShape[newShape.size() - 2] == 1)
	{
		newShape.erase(std::prev(newShape.cend()));
	}

	return newShape;
}

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

bool isShapeExtendableToAnother(const std::vector<size_t>& shape, const std::vector<size_t>& targetShape)
{
	if(targetShape.size() < shape.size())
	{
		return false;
	}

	return std::equal(shape.crbegin(), shape.crend(), targetShape.crbegin());
}
} // namespace mlCore::detail