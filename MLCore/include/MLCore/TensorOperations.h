#include <MLCore/BasicTensor.h>
#include <cmath>

namespace mlCore
{
/**
 * @brief Set of both binary and unary operators for tensors 
 * 
 */
struct TensorOperations
{
	template <typename valueType>
	static BasicTensor<valueType> power(const BasicTensor<valueType>& lhs,
										const BasicTensor<valueType>& rhs)
	{
		if(lhs.shape_ == rhs.shape_)
		{
			BasicTensor<valueType> ret(lhs.shape_);
			for(size_t i = 0; i < lhs.length_; i++)
				ret.data_[i] = pow(lhs.data_[i], rhs.data_[i]);
			return ret;
		}

		static const std::function<valueType(const valueType l, const valueType r)> powOperator_ =
			[](const valueType l, const valueType r) { return pow(l, r); };

		return lhs.performOperation_(rhs, powOperator_);
	}

	template <typename valueType>
	static BasicTensor<valueType> ln(const BasicTensor<valueType>& arg)
	{
		auto ret = arg;
		for(auto& val : ret)
			val = log(val);
		return ret;
	}

	template <typename valueType>
	static BasicTensor<valueType> relu(const BasicTensor<valueType>& arg)
	{
		auto ret = arg;
		for(auto& val : ret)
			val = val > 0 ? val : 0;
		return ret;
	}

	template <typename valueType>
	static BasicTensor<valueType> sigmoid(const BasicTensor<valueType>& arg)
	{
		auto ret = arg;
		for(auto& val : ret)
			val = 1.0 / (1.0 + pow(M_E, -val));
		return ret;
	}
};
} // namespace mlCore