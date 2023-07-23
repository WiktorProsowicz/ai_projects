#include <AutoDiff/BinaryOperators/MultiplyOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
void MultiplyOperator::updateValue()
{
	value_ = lhsInput_->getValue() * rhsInput_->getValue();
}
} // namespace mlCore::autoDiff::binaryOperators