#include "AutoDiff/UnaryOperators/ReluOperator.h"

#include "MLCore/TensorOperations.h"

namespace mlCore::autoDiff::unaryOperators
{
void ReluOperator::updateValue()
{
	value_ = TensorOperations::relu(input_->getValue());
}

Tensor ReluOperator::computeDerivative(const Tensor& outerDerivative) const
{

	return computeDirectDerivative() * outerDerivative;
}

Tensor ReluOperator::computeDirectDerivative() const
{
	auto inputCopy = input_->getValue();

	for(auto& val : inputCopy)
	{
		val = val > 0 ? 1 : 0;
	}

	return inputCopy;
}
} // namespace mlCore::autoDiff::unaryOperators
