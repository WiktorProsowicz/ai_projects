#include <AutoDiff/BinaryOperators/MatmulOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
void MatmulOperator::updateValue()
{
	value_ = lhsInput_->getValue().matmul(rhsInput_->getValue());
}
} // namespace mlCore::autoDiff::binaryOperators