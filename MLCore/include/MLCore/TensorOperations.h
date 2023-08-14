#include <MLCore/BasicTensor.h>
#include <cmath>

namespace mlCore
{
/**
 * @brief Set of both binary and unary operators for tensors 
 * 
 */
template <typename ValueType>
class BasicTensorOperations
{
public:
	static BasicTensor<ValueType> power(const BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs);

	static BasicTensor<ValueType> ln(const BasicTensor<ValueType>& arg);

	static BasicTensor<ValueType> relu(const BasicTensor<ValueType>& arg);

	static BasicTensor<ValueType> sigmoid(const BasicTensor<ValueType>& arg);
};

using TensorOperations = BasicTensorOperations<double>;
} // namespace mlCore