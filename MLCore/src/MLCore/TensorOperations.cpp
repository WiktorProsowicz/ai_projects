#include <MLCore/TensorOperations.h>
#include <MLCore/TensorOperationsImpl.h>
#include <cmath>

namespace mlCore
{
template class BasicTensorOperations<double>;

template <typename ValueType>
BasicTensor<ValueType> BasicTensorOperations<ValueType>::power(const BasicTensor<ValueType>& lhs,
															   const BasicTensor<ValueType>& rhs)
{
	auto ret = lhs;
	TensorOperationsImpl<ValueType>::powerInPlace(ret, rhs);
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
} // namespace mlCore