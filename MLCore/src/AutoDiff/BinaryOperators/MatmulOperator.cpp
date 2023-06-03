#include <AutoDiff/BinaryOperators/MatmulOperator.h>

namespace mlCore
{
void MatmulOperator::updateValue()
{
	value_ = lhsInput_->getValue().matmul(rhsInput_->getValue());
}
} // namespace mlCore