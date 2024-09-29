#include "MLCore/TensorOperations.h"

#include <cmath>
#include <cstddef>
#include <iterator>
#include <map>
#include <vector>

#include <LoggingLib/LoggingLib.hpp>
#include <fmt/core.h>

#include "MLCore/TensorOperationsImpl.h"
#include "MLCore/Utilities.h"
#include "MLCore/UtilitiesImpl.h"

namespace mlCore
{
template class BasicTensorOperations<double>;

template <typename ValueType>
BasicTensor<ValueType> BasicTensorOperations<ValueType>::power(const BasicTensor<ValueType>& lhs,
															   const BasicTensor<ValueType>& rhs)
{
	auto ret = lhs;
	detail::TensorOperationsImpl<ValueType>::powerInPlace(ret, rhs);
	return ret;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensorOperations<ValueType>::ln(const BasicTensor<ValueType>& arg)
{
	auto ret = arg;
	for(auto& val : ret)
	{
		val = std::log(val);
	}
	return ret;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensorOperations<ValueType>::relu(const BasicTensor<ValueType>& arg)
{
	auto ret = arg;
	for(auto& val : ret)
	{
		val = val > 0 ? val : 0;
	}
	return ret;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensorOperations<ValueType>::sigmoid(const BasicTensor<ValueType>& arg)
{
	auto ret = arg;
	for(auto& val : ret)
	{
		val = 1.0 / (1.0 + std::pow(M_E, -val));
	}
	return ret;
}

namespace detail
{
/// Traverses TensorForm extracting data from it.
template <typename TensorDataType>
class TensorFormVisitor
{
public:
	TensorFormVisitor() = default;

	std::vector<TensorDataType> operator()(const TensorDataType& singleValue)
	{
		return {singleValue};
	}

	std::vector<TensorDataType> operator()(const RawTensorInitList<TensorDataType>& containerValue)
	{

		if(_collectedShapeIndices.find(_currentLevel) == _collectedShapeIndices.end())
		{
			_collectedShapeIndices.emplace(_currentLevel, containerValue.size());
		}
		else if(containerValue.size() != _collectedShapeIndices.at(_currentLevel))
		{
			LOG_ERROR("TensorOperations",
					  fmt::format("Inconsistent elements number at axis {}.", _currentLevel));
		}

		std::vector<std::vector<TensorDataType>> collectedValueSets;
		collectedValueSets.reserve(containerValue.size());

		_currentLevel++;

		for(const auto& valueSet : containerValue)
		{
			collectedValueSets.emplace_back(std::visit(*this, valueSet));
		}

		_currentLevel--;

		if(collectedValueSets.empty())
		{
			LOG_ERROR("TensorOperations",
					  "Encountered empty initializer list at a certain level of raw tensor form.");
		}

		const auto nElementsInSubValue = collectedValueSets.cbegin()->size();

		std::vector<TensorDataType> collectedValues;
		collectedValues.reserve(collectedValueSets.size() * nElementsInSubValue);

		for(const auto& valueSet : collectedValueSets)
		{
			if(valueSet.size() != nElementsInSubValue)
			{
				LOG_ERROR(
					"TensorOperations",
					"Encountered not-constant number of subelements at a certain level of raw tensor form.");
			}

			for(const auto& value : valueSet)
			{
				collectedValues.push_back(value);
			}
		}

		return collectedValues;
	}

	std::vector<size_t> getShape() const
	{
		std::vector<size_t> collectedShape;
		collectedShape.reserve(_collectedShapeIndices.size());

		std::transform(_collectedShapeIndices.cbegin(),
					   _collectedShapeIndices.cend(),
					   std::back_inserter(collectedShape),
					   [](const auto& axisIndex) { return axisIndex.second; });

		return collectedShape;
	}

private:
	std::map<size_t, size_t> _collectedShapeIndices{};
	size_t _currentLevel = 0;
};
} // namespace detail

template <typename ValueType>
BasicTensor<ValueType> BasicTensorOperations<ValueType>::makeTensor(const TensorForm<ValueType>& tensorForm)
{
	detail::TensorFormVisitor<ValueType> visitor;

	const auto collectedValues = std::visit(visitor, tensorForm);
	const auto shape = visitor.getShape();

	mlCore::BasicTensor<ValueType> tensor(shape);

	tensor.fill(collectedValues.begin(), collectedValues.end());

	return tensor;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensorOperations<ValueType>::transpose(const BasicTensor<ValueType>& arg,
																   MatrixSpec spec)
{
	const auto argShape = detail::applyMatSpecToShape(arg.shape(), spec);

	const size_t frameShapeFirst = *(++argShape.rbegin());
	const size_t frameShapeSecond = *argShape.rbegin();
	// Size of a single 2-dimensional part that takes part in single matrix multiplication
	const size_t frameLength = frameShapeFirst * frameShapeSecond;

	std::vector<size_t> retShape = argShape;
	std::swap(retShape[retShape.size() - 1], retShape[retShape.size() - 2]);

	BasicTensor<ValueType> ret(retShape);

	for(size_t frameOffset = 0; frameOffset < arg._length; frameOffset += frameLength)
	{
		for(size_t posInFrame = 0; posInFrame < frameLength; posInFrame++)
		{
			ret._data[frameOffset + posInFrame] =
				arg._data[frameOffset + (posInFrame % frameShapeFirst) * frameShapeSecond +
						  (posInFrame / frameShapeFirst)];
		}
	}

	return ret;
}

namespace
{
/**
 * @brief Performs matrix multiplication of two matrices with given shapes and data pointers.
 *
 * @details The algorithm may be used multiple times over a bigger chunk of data if the tensors being
 * multiplied have more than 2 dimensions.
 *
 * @param lhsData Pointer to the lhs matrix' data.
 * @param rhsData Pointer to the rhs matrix' data.
 * @param resData Pointer to the result matrix' data.
 * @param lhsRows Number of rows in the lhs matrix.
 * @param lhsCols Number of columns in the lhs matrix.
 * @param rhsCols Number of columns in the rhs matrix.
 */
template <typename ValueType>
void performSingleMatmul(const ValueType* const lhsData,
						 const ValueType* const rhsData,
						 ValueType* const resData,
						 const size_t lhsRows,
						 const size_t lhsCols,
						 const size_t rhsCols)
{
	for(size_t rowIter = 0; rowIter < lhsRows; rowIter++)
	{
		for(size_t colIter = 0; colIter < rhsCols; colIter++)
		{
			for(size_t mulIter = 0; mulIter < lhsCols; mulIter++)
			{
				resData[rowIter * rhsCols + colIter] +=
					lhsData[rowIter * lhsCols + mulIter] * rhsData[mulIter * rhsCols + colIter];
			}
		}
	}
}

/**
 * @brief Performs multiplication between tensors one of which is a classical matrix.
 *
 * @param lhsData Pointer to the lhs matrix' data.
 * @param lhsShape Shape of the lhs matrix.
 * @param rhsData Pointer to the rhs tensor's data.
 * @param rhsShape Shape of the rhs tensor.
 * @param resData Pointer to the result tensor's data.
 */
template <typename ValueType>
void performMatmulWithNormalMatrix(const ValueType* const lhsData,
								   const std::vector<size_t>& lhsShape,
								   const ValueType* const rhsData,
								   const std::vector<size_t>& rhsShape,
								   ValueType* const resData)
{
	const auto& lhsRows = lhsShape[lhsShape.size() - 2];
	const auto& lhsCols = lhsShape[lhsShape.size() - 1];
	const auto& rhsCols = rhsShape[rhsShape.size() - 1];

	if(lhsShape.size() == 2)
	{
		const auto rhsLength =
			std::accumulate(rhsShape.cbegin(), rhsShape.cend(), size_t{1}, std::multiplies<>());

		for(size_t rhsPos = 0, resPos = 0; rhsPos < rhsLength;
			rhsPos += rhsCols * lhsCols, resPos += rhsCols * lhsRows)
		{
			performSingleMatmul(lhsData, rhsData + rhsPos, resData + resPos, lhsRows, lhsCols, rhsCols);
		}

		return;
	}

	const auto lhsLength =
		std::accumulate(lhsShape.cbegin(), lhsShape.cend(), size_t{1}, std::multiplies<>());

	for(size_t lhsPos = 0, resPos = 0; lhsPos < lhsLength;
		lhsPos += lhsCols * lhsRows, resPos += lhsRows * rhsCols)
	{
		performSingleMatmul(lhsData + lhsPos, rhsData, resData + resPos, lhsRows, lhsCols, rhsCols);
	}
}

template <typename ValueType>
void performMatmulWithBroadcastedTensors(const ValueType* const lhsData,
										 const std::vector<size_t>& lhsPaddedShape,
										 const ValueType* const rhsData,
										 const std::vector<size_t>& rhsPaddedShape,
										 ValueType* const resData)
{
	// Tells the position of a single computed matrix relative to the array of values
	auto computeFramePos = [](const std::vector<size_t>& treePath, const std::vector<size_t>& shape) -> size_t
	{
		size_t offset = 0;
		size_t factor = shape[shape.size() - 1] * shape[shape.size() - 2];
		for(size_t i = shape.size() - 3; i < shape.size() - 2; i--)
		{
			offset += treePath[i] * factor;
			factor *= shape[i];
		}
		return offset;
	};

	const auto& shapeSize = lhsPaddedShape.size();
	const auto& lhsCols = lhsPaddedShape[shapeSize - 1];
	const auto& rhsCols = rhsPaddedShape[shapeSize - 1];
	const auto& lhsRows = lhsPaddedShape[shapeSize - 2];

	std::vector<size_t> lhsTreePath(shapeSize - 2, 0);
	std::vector<size_t> rhsTreePath(shapeSize - 2, 0);
	const auto resTensorLength =
		std::accumulate(lhsPaddedShape.cbegin(), lhsPaddedShape.cend(), size_t{1}, std::multiplies<>());

	for(size_t resElementPos = 0; resElementPos < resTensorLength; resElementPos += (lhsRows * rhsCols))
	{
		const ValueType* lhsDataPtr = lhsData + computeFramePos(lhsTreePath, lhsPaddedShape);
		const ValueType* rhsDataPtr = rhsData + computeFramePos(rhsTreePath, rhsPaddedShape);

		performSingleMatmul(lhsDataPtr, rhsDataPtr, resData + resElementPos, lhsRows, lhsCols, rhsCols);

		// Updating tree paths
		for(size_t i = shapeSize - 3; i < shapeSize - 2; i--)
		{

			if(lhsPaddedShape[i] > 1)
			{
				lhsTreePath[i]++;
			}

			if(rhsPaddedShape[i] > 1)
			{
				rhsTreePath[i]++;
			}

			if((lhsTreePath[i] < lhsPaddedShape[i]) && (lhsTreePath[i] < rhsPaddedShape[i]))
			{
				break;
			}

			lhsTreePath[i] = 0;
			rhsTreePath[i] = 0;
		}
	}
}
} // namespace

template <typename ValueType>
BasicTensor<ValueType> BasicTensorOperations<ValueType>::matmul(const BasicTensor<ValueType>& lhs,
																const BasicTensor<ValueType>& rhs,
																MatrixSpec lhsSpec,
																MatrixSpec rhsSpec)
{
	const auto lhsShape = detail::applyMatSpecToShape(lhs.shape(), lhsSpec);
	const auto rhsShape = detail::applyMatSpecToShape(rhs.shape(), rhsSpec);

	detail::assertCanMatmulTensors(lhsShape, rhsShape);

	const auto [lhsPaddedShape, rhsPaddedShape] = detail::padShapes(lhsShape, rhsShape);

	BasicTensor<ValueType> resultTensor(detail::getReturnShapeForMatmul(lhsPaddedShape, rhsPaddedShape), 0);

	if(lhsShape.size() == 2 || rhsShape.size() == 2)
	{
		performMatmulWithNormalMatrix(lhs._data, lhsShape, rhs._data, rhsShape, resultTensor._data);
		return resultTensor;
	}

	return resultTensor;
}

template <typename ValueType>
BasicTensor<ValueType> BasicTensorOperations<ValueType>::reduceAdd(const BasicTensor<ValueType>& arg,
																   const std::vector<size_t>& targetShape)
{
	if(!detail::isShapeExtendableToAnother(targetShape, arg.shape()))
	{
		LOG_ERROR("TensorOperations",
				  fmt::format("Cannot reduce-add tensor. The shape {} is not reducable to the target "
							  "tensor's shape {}!",
							  detail::stringifyVector(arg.shape()),
							  detail::stringifyVector(targetShape)));
	}

	BasicTensor<ValueType> ret(targetShape, 0);

	for(size_t framePos = 0; framePos < arg._length; framePos += ret._length)
	{
		for(size_t i = 0; i < ret._length; i++)
		{
			ret._data[i] += arg._data[framePos + i];
		}
	}

	return ret;
}

namespace
{
/// @brief Tells if two given tensors have the same shape except the given axis.
template <typename ValueType>
bool canStackTensors(const BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs, const size_t axis)
{
	if(lhs.shape().size() != rhs.shape().size())
	{
		return false;
	}

	for(size_t i = 0; i < lhs.shape().size(); i++)
	{
		if(i == axis)
		{
			continue;
		}

		if(lhs.shape()[i] != rhs.shape()[i])
		{
			return false;
		}
	}

	return true;
}
} // namespace

template <typename ValueType>
BasicTensor<ValueType>
BasicTensorOperations<ValueType>::stack(const std::vector<BasicTensor<ValueType>>& tensors, const size_t axis)
{
	if(tensors.empty())
	{
		LOG_ERROR("TensorOperations", "Cannot stack empty list of tensors!");
	}

	if(const auto unstackableIt = std::adjacent_find(tensors.cbegin(),
													 tensors.cend(),
													 [axis](const auto& lhs, const auto& rhs)
													 { return !canStackTensors(lhs, rhs, axis); });
	   unstackableIt != tensors.cend())
	{
		LOG_ERROR("TensorOperations",
				  fmt::format("Cannot stack tensors at axis {}. Incompatible shapes: {} and {}",
							  axis,
							  detail::stringifyVector(unstackableIt->shape()),
							  detail::stringifyVector(std::next(unstackableIt)->shape())));
	}

	mlCore::TensorShape retShape = tensors.front().shape();
	retShape[axis] =
		std::accumulate(tensors.cbegin(),
						tensors.cend(),
						size_t{0},
						[axis](const auto& acc, const auto& tensor) { return acc + tensor.shape()[axis]; });

	BasicTensor<ValueType> ret(retShape);

	std::vector<size_t> frameSizes;
	frameSizes.reserve(tensors.size());

	std::transform(tensors.cbegin(),
				   tensors.cend(),
				   std::back_inserter(frameSizes),
				   [axis](const auto& tensor)
				   {
					   return std::accumulate(tensor.shape().cbegin() + axis,
											  tensor.shape().cend(),
											  size_t{1},
											  std::multiplies<>());
				   });

	const size_t stackingIterations =
		std::accumulate(retShape.cbegin(), retShape.cbegin() + axis, size_t{1}, std::multiplies<>());

	ValueType* retData = ret._data;
	for(size_t i = 0; i < stackingIterations; i++)
	{
		for(size_t j = 0; j < tensors.size(); j++)
		{
			const auto& tensor = tensors[j];
			const auto& frameSize = frameSizes[j];

			std::copy(tensor._data + i * frameSize, tensor._data + (i + 1) * frameSize, retData);

			retData += frameSize;
		}
	}

	return ret;
}
} // namespace mlCore
