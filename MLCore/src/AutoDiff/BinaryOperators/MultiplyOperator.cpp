#include <AutoDiff/BinaryOperators/MultiplyOperator.h>

namespace mlCore
{
void MultiplyOperator::updateValue()
{
	value_ = lhsInput_->getValue() * rhsInput_->getValue();
}
} // namespace mlCore