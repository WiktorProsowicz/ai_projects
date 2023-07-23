#include <AutoDiff/BinaryOperators/SubtractOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
void SubtractOperator::updateValue()
{
	value_ = lhsInput_->getValue() - rhsInput_->getValue();
}
} // namespace mlCore::autoDiff::binaryOperators