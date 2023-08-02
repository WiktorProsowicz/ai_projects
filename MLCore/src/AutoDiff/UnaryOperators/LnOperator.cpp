#include <AutoDiff/UnaryOperators/LnOperator.h>
#include <MLCore/TensorOperations.h>

namespace mlCore::autoDiff::unaryOperators
{
void LnOperator::updateValue()
{
	value_ = TensorOperations::ln(input_->getValue());
}
} // namespace mlCore::autoDiff::unaryOperators