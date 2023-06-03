#include <AutoDiff/UnaryOperators/SigmoidOperator.h>
#include <MLCore/TensorOperations.h>

namespace mlCore
{
void SigmoidOperator::updateValue()
{
	value_ = TensorOperations::sigmoid(input_->getValue());
}
} // namespace mlCore