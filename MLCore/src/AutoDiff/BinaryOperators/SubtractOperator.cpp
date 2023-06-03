#include <AutoDiff/BinaryOperators/SubtractOperator.h>

namespace mlCore
{
void SubtractOperator::updateValue()
{
	value_ = lhsInput_->getValue() - rhsInput_->getValue();
}
} // namespace mlCore