#include <AutoDiff/BinaryOperators/PowerOperator.h>
#include <MLCore/TensorOperations.h>

namespace mlCore::autoDiff::binaryOperators
{
void PowerOperator::updateValue()
{
	value_ = TensorOperations::power(lhsInput_->getValue(), rhsInput_->getValue());
}
} // namespace mlCore::autoDiff::binaryOperators