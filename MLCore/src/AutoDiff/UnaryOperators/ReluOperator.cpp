#include <AutoDiff/UnaryOperators/ReluOperator.h>
#include <MLCore/TensorOperations.h>

namespace mlCore::autoDiff::unaryOperators
{
void ReluOperator::updateValue()
{
	value_ = TensorOperations::relu(input_->getValue());
}
} // namespace mlCore::autoDiff::unaryOperators