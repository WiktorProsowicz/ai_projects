#ifndef MLCORE_SRC_INCLUDE_MLCORE_TENSOROPERATIONSIMPL_H
#define MLCORE_SRC_INCLUDE_MLCORE_TENSOROPERATIONSIMPL_H

#include <MLCore/BasicTensor.h>

namespace mlCore
{

template <typename ValueType>
class TensorOperationsImpl
{
public:
	/// Adds right tensor to the left one
	static void addTensorsInPlace(BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs);

	/// Multiplies left tensor by the right one
	static void multiplyTensorsInPlace(BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs);

	/// Subtracts right tensor from the left one
	static void subtractTensorsInPlace(BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs);

	/// Divides left tensor by the right one
	static void divideTensorsInPlace(BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs);

	/// Computes left tensor to the power of right one
	static void powerInPlace(BasicTensor<ValueType>& lhs, const BasicTensor<ValueType>& rhs);
};
} // namespace mlCore

#endif