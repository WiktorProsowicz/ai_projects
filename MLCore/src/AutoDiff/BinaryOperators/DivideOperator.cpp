#include <AutoDiff/BinaryOperators/DivideOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
void DivideOperator::updateValue()
{
	value_ = lhsInput_->getValue() / rhsInput_->getValue();
}
} // namespace mlCore::autoDiff::binaryOperators