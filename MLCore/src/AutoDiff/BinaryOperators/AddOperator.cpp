#include <AutoDiff/BinaryOperators/AddOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
void AddOperator::updateValue()
{
	value_ = lhsInput_->getValue() + rhsInput_->getValue();
}
} // namespace mlCore::autoDiff::binaryOperators