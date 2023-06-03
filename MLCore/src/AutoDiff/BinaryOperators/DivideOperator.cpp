#include <AutoDiff/BinaryOperators/DivideOperator.h>

namespace mlCore
{
void DivideOperator::updateValue()
{
	value_ = lhsInput_->getValue() / rhsInput_->getValue();
}
} // namespace mlCore