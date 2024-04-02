#include "AutoDiff/UnaryOperators/ReluOperator.h"

#include "MLCore/TensorOperations.h"

namespace mlCore::autoDiff::unaryOperators
{
void ReluOperator::updateValue()
{
	_value = TensorOperations::relu(_input->getValue());
}

Tensor ReluOperator::computeDerivative(const Tensor& outerDerivative) const
{

	return computeDirectDerivative() * outerDerivative;
}

Tensor ReluOperator::computeDirectDerivative() const
{
	auto inputCopy = _input->getValue();

	for(auto& val : inputCopy)
	{
		val = val > 0 ? 1 : 0;
	}

	return inputCopy;
}
} // namespace mlCore::autoDiff::unaryOperators
