#include "MLCore/TensorOperations.h"

#include <cmath>

#include "MLCore/TensorOperationsImpl.h"

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

		for(const auto& [axis, index] : _collectedShapeIndices)
		{
			collectedShape.push_back(index);
		}

		return collectedShape;
	}

private:
	std::map<size_t, size_t> _collectedShapeIndices;
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
} // namespace mlCore
