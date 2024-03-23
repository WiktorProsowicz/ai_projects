#include <MLCore/TensorOperationsImpl.h>

#include <cmath>

#include <fmt/format.h>

#include <MLCore/Utilities.h>
#include <MLCore/BasicTensor.h>

// NOLINTBEGIN

#define __SIMPLE_PLUS(a, b) a + b
#define __SIMPLE_MINUS(a, b) a - b
#define __SIMPLE_DIVIDE(a, b) a / b
#define __SIMPLE_MULTIPLY(a, b) a* b

// Computes operation between two tensors with same shape and assigns result to the left one
#define COMPAT_SHAPES_OPERATION(lhs, rhs, oper)                                                                                  \
	if(lhs.shape_ == rhs.shape_)                                                                                                 \
	{                                                                                                                            \
		for(size_t dataPos = 0; dataPos < lhs.length_; dataPos++)                                                                \
		{                                                                                                                        \
			lhs.data_[dataPos] = oper(lhs.data_[dataPos], rhs.data_[dataPos]);                                                   \
		}                                                                                                                        \
		return;                                                                                                                  \
	}

// Used when one of the tensors is scalar type and expensive broadcasting could be avoided
#define OPERATION_WITH_SCALAR(lhs, rhs, oper)                                                                                    \
	if(rhs.shape_.empty())                                                                                                       \
	{                                                                                                                            \
		for(size_t dataPos = 0; dataPos < lhs.length_; dataPos++)                                                                \
		{                                                                                                                        \
			lhs.data_[dataPos] = oper(lhs.data_[dataPos], rhs.data_[0]);                                                         \
		}                                                                                                                        \
		return;                                                                                                                  \
	}

// Computes the position of an element in tensor's payload
#define COMPUTE_ELEMENT_POSITION(path, shape, positionName)                                                                      \
	size_t positionName = 0;                                                                                                     \
	{                                                                                                                            \
		size_t offset = 1;                                                                                                       \
		for(size_t i = path.size() - 1; i < path.size(); i--)                                                                    \
		{                                                                                                                        \
			positionName += offset * path[i];                                                                                    \
			offset *= shape[i];                                                                                                  \
		}                                                                                                                        \
	}

// Stretches a tensor and applies an operation
#define STRETCH_TENSOR_TO_ANOTHER(dst, src, srcPaddedShape, oper)                                                                \
	std::vector<size_t> dstTreePath(dst.nDimensions(), 0);                                                                       \
	std::vector<size_t> srcTreePath(dst.nDimensions(), 0);                                                                       \
                                                                                                                                 \
	size_t elementsProcessed = 0;                                                                                                \
                                                                                                                                 \
	while(elementsProcessed < dst.length_)                                                                                       \
	{                                                                                                                            \
		COMPUTE_ELEMENT_POSITION(dstTreePath, dst.shape_, dstElemPos)                                                            \
		COMPUTE_ELEMENT_POSITION(srcTreePath, srcPaddedShape, srcElemPos)                                                        \
                                                                                                                                 \
		dst.data_[dstElemPos] = oper(dst.data_[dstElemPos], src.data_[srcElemPos]);                                              \
                                                                                                                                 \
		elementsProcessed++;                                                                                                     \
		for(size_t i = 0; i < dst.nDimensions(); i++)                                                                            \
		{                                                                                                                        \
			const size_t pathPos = dst.nDimensions() - 1 - i;                                                                    \
                                                                                                                                 \
			dstTreePath[pathPos]++;                                                                                              \
                                                                                                                                 \
			if(srcPaddedShape[pathPos] > 1)                                                                                      \
			{                                                                                                                    \
				srcTreePath[pathPos]++;                                                                                          \
			}                                                                                                                    \
                                                                                                                                 \
			if(dstTreePath[pathPos] < dst.shape_[pathPos])                                                                       \
			{                                                                                                                    \
				break;                                                                                                           \
			}                                                                                                                    \
                                                                                                                                 \
			dstTreePath[pathPos] = 0;                                                                                            \
			srcTreePath[pathPos] = 0;                                                                                            \
		}                                                                                                                        \
	}

// Generic operation between tensors with incompatible shapes - result assigned to the left one
#define BROADCASTED_TENSOR_OPERATION(lhs, rhs, oper)                                                                             \
	checkShapesForBroadcasting(lhs.shape_, rhs.shape_);                                                                          \
                                                                                                                                 \
	const auto biggerSize = std::max(lhs.shape_.size(), rhs.shape_.size());                                                      \
                                                                                                                                 \
	const auto paddedLeftShape = padShapeFromLeft(lhs.shape_, biggerSize);                                                       \
	const auto paddedRightShape = padShapeFromLeft(rhs.shape_, biggerSize);                                                      \
                                                                                                                                 \
	const auto retShape = deduceBroadcastedShape(paddedLeftShape, paddedRightShape);                                             \
                                                                                                                                 \
	BasicTensor<ValueType> ret(retShape, 0.0);                                                                                   \
                                                                                                                                 \
	{                                                                                                                            \
		STRETCH_TENSOR_TO_ANOTHER(ret, lhs, paddedLeftShape, __SIMPLE_PLUS);                                                     \
	}                                                                                                                            \
                                                                                                                                 \
	{                                                                                                                            \
		STRETCH_TENSOR_TO_ANOTHER(ret, rhs, paddedRightShape, oper);                                                             \
	}                                                                                                                            \
                                                                                                                                 \
	lhs = std::move(ret);

namespace mlCore::detail
{
namespace
{
/**
 * @brief Checks if the shapes of two tensors can be stretched to perform the broadcast operation
 *
 * @param shape1
 * @param shape2
 */
void checkShapesForBroadcasting(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2)
{
	// checking if the rules of broadcasting are not breached
	for(auto [leftShapeIter, rightShapeIter] = std::tuple{shape1.crbegin(), shape2.crbegin()};
		(leftShapeIter > shape1.crend()) && (rightShapeIter > shape2.crend());
		leftShapeIter--, rightShapeIter--)
	{
		if((*leftShapeIter != 1) && (*rightShapeIter != 1) && (*leftShapeIter != *rightShapeIter))
		{
			LOG_ERROR("TensorOperations",
					  fmt::format("Cannot perform broadcasting operation on tensors with invalid shapes: '{}' '{}'.",
								  stringifyVector(shape1),
								  stringifyVector(shape2)));
		}
	}
}

/**
 * @brief Fills tensor's shape from left side with given value
 *
 * @param shape Shape to be filled
 * @param biggerSize Size of the result tensor. If less than shape.size(), no padding is added
 * @param paddingVal Value to pad the shape with
 * @return Padded vector
 */
std::vector<size_t> padShapeFromLeft(const std::vector<size_t>& shape, const size_t biggerSize, const size_t paddingVal = 1)
{
	std::vector<size_t> paddedShape(biggerSize, paddingVal);

	std::copy(shape.cbegin(), shape.cend(), std::next(paddedShape.begin(), static_cast<ptrdiff_t>(biggerSize - shape.size())));

	return paddedShape;
}

std::vector<size_t> deduceBroadcastedShape(const std::vector<size_t>& paddedShape1, const std::vector<size_t>& paddedShape2)
{

	std::vector<size_t> retShape(paddedShape1.size());
	for(size_t shapePos = 0; shapePos < paddedShape1.size(); shapePos++)
	{
		// filling new shape according to dimension that can be stretched
		retShape[shapePos] = paddedShape1[shapePos] == 1 ? paddedShape2[shapePos] : paddedShape1[shapePos];
	}

	return retShape;
}

} // namespace

template class TensorOperationsImpl<double>;

template <typename ValueType>
void TensorOperationsImpl<ValueType>::addTensorsInPlace(BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs)
{
	COMPAT_SHAPES_OPERATION(lhs, rhs, __SIMPLE_PLUS)
	OPERATION_WITH_SCALAR(lhs, rhs, __SIMPLE_PLUS)
	BROADCASTED_TENSOR_OPERATION(lhs, rhs, __SIMPLE_PLUS)
}

template <typename ValueType>
void TensorOperationsImpl<ValueType>::multiplyTensorsInPlace(BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs)
{
	COMPAT_SHAPES_OPERATION(lhs, rhs, __SIMPLE_MULTIPLY)
	OPERATION_WITH_SCALAR(lhs, rhs, __SIMPLE_MULTIPLY)
	BROADCASTED_TENSOR_OPERATION(lhs, rhs, __SIMPLE_MULTIPLY)
}

template <typename ValueType>
void TensorOperationsImpl<ValueType>::subtractTensorsInPlace(BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs)
{
	COMPAT_SHAPES_OPERATION(lhs, rhs, __SIMPLE_MINUS)
	OPERATION_WITH_SCALAR(lhs, rhs, __SIMPLE_MINUS)
	BROADCASTED_TENSOR_OPERATION(lhs, rhs, __SIMPLE_MINUS)
}

template <typename ValueType>
void TensorOperationsImpl<ValueType>::divideTensorsInPlace(BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs)
{
	COMPAT_SHAPES_OPERATION(lhs, rhs, __SIMPLE_DIVIDE)
	OPERATION_WITH_SCALAR(lhs, rhs, __SIMPLE_DIVIDE)
	BROADCASTED_TENSOR_OPERATION(lhs, rhs, __SIMPLE_DIVIDE)
}

template <typename ValueType>
void TensorOperationsImpl<ValueType>::powerInPlace(BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs)
{
	COMPAT_SHAPES_OPERATION(lhs, rhs, std::pow)
	OPERATION_WITH_SCALAR(lhs, rhs, std::pow)
	BROADCASTED_TENSOR_OPERATION(lhs, rhs, std::pow)
}

} // namespace mlCore::detail

// NOLINTEND
