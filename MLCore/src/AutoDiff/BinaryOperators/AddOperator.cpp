#include <AutoDiff/BinaryOperators/AddOperator.h>

namespace mlCore
{
void AddOperator::updateValue()
{
	value_ = lhsInput_->getValue() + rhsInput_->getValue();
}
} // namespace mlCore