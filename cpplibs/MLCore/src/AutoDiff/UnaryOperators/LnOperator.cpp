#include "AutoDiff/UnaryOperators/LnOperator.h"

#include "MLCore/TensorOperations.h"

namespace mlCore::autoDiff::unaryOperators
{
void LnOperator::updateValue()
{
	_value = TensorOperations::ln(_input->getValue());
}

Tensor LnOperator::computeDerivative(const Tensor& outerDerivative) const
{
	return computeDirectDerivative() * outerDerivative;
}

Tensor LnOperator::computeDirectDerivative() const
{
	auto inputCopy = _input->getValue();

	for(auto& val : inputCopy)
	{
		val = 1.0 / val;
	}

	return inputCopy;
}
} // namespace mlCore::autoDiff::unaryOperators
