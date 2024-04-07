#include "MLCore/Utilities.h"

#include <stdexcept>
#include <vector>

namespace mlCore::detail
{
std::vector<size_t> getOutputShapeForMatmul(const std::vector<size_t>& lhsShape,
											const std::vector<size_t>& rhsShape)
{
	// checking if first tensor can be padded to nDims >= 2
	if(rhsShape.size() < 2)
	{
		throw std::runtime_error("Failed to compose output shape of matrix multiplication. Cannot obtain "
								 "last but one dimension of the second tensor!");
	}

	// for padding tensors shapes
	const size_t biggerSize = std::max(lhsShape.size(), rhsShape.size());

	// padded shapes for easier operating
	std::vector<size_t> paddedShapeFirst(biggerSize, 1);
	std::vector<size_t> paddedShapeSecond(biggerSize, 1);

	std::copy(lhsShape.cbegin(),
			  lhsShape.cend(),
			  std::next(paddedShapeFirst.begin(), static_cast<ptrdiff_t>(biggerSize - lhsShape.size())));

	std::copy(rhsShape.cbegin(),
			  rhsShape.cend(),
			  std::next(paddedShapeSecond.begin(), static_cast<ptrdiff_t>(biggerSize - rhsShape.size())));

	// checking matmul conditions
	if(paddedShapeFirst[biggerSize - 1] != paddedShapeSecond[biggerSize - 2])
	{
		throw std::runtime_error(
			"Failed to compose output shape of matrix multiplication. Last two dimensions are incompatible!");
	}

	for(size_t i = 0; i < biggerSize - 2; i++)
	{
		if((paddedShapeFirst[i] != paddedShapeSecond[i]) && (paddedShapeFirst[i] != 1) &&
		   (paddedShapeSecond[i] != 1))
		{
			throw std::runtime_error(
				"Failed to compose output shape of matrix multiplication. Shapes are incompatible!");
		}
	}

	std::vector<size_t> retShape(biggerSize);
	retShape[biggerSize - 2] = paddedShapeFirst[biggerSize - 2];
	retShape[biggerSize - 1] = paddedShapeSecond[biggerSize - 1];

	for(size_t i = 0; i < biggerSize - 2; i++)
	{
		retShape[i] = paddedShapeFirst[i] == 1 ? paddedShapeSecond[i] : paddedShapeFirst[i];
	}

	return retShape;
}
} // namespace mlCore::detail