#include <AutoDiff/BinaryOperators/PowerOperator.h>
#include <MLCore/TensorOperations.h>

namespace mlCore
{
void PowerOperator::updateValue()
{
	value_ = TensorOperations::power(lhsInput_->getValue(), rhsInput_->getValue());
}
} // namespace mlCore