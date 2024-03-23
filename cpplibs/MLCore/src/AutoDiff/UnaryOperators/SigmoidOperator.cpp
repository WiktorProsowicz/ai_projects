#include "AutoDiff/UnaryOperators/SigmoidOperator.h"

#include "MLCore/TensorOperations.h"

namespace mlCore::autoDiff::unaryOperators
{
void SigmoidOperator::updateValue()
{
	value_ = TensorOperations::sigmoid(input_->getValue());
}

Tensor SigmoidOperator::computeDerivative(const Tensor& outerDerivative) const
{

	return computeDirectDerivative() * outerDerivative;
}

Tensor SigmoidOperator::computeDirectDerivative() const
{
	auto valueCopy = value_;

	for(auto& val : valueCopy)
	{
		val = val * (1 - val);
	}

	return valueCopy;
}
} // namespace mlCore::autoDiff::unaryOperators
