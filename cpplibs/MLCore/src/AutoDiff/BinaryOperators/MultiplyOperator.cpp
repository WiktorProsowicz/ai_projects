#include "AutoDiff/BinaryOperators/MultiplyOperator.h"

namespace mlCore::autoDiff::binaryOperators
{
void MultiplyOperator::updateValue()
{
	_value = _lhsInput->getValue() * _rhsInput->getValue();
}

std::pair<Tensor, Tensor> MultiplyOperator::computeDerivative(const Tensor& outerDerivative) const
{
	auto [leftDerivative, rightDerivative] = computeDirectDerivative();

	return {leftDerivative * outerDerivative, rightDerivative * outerDerivative};
}

std::pair<Tensor, Tensor> MultiplyOperator::computeDirectDerivative() const
{
	const auto& [leftInputNode, rightInputNode] = getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	return {rightValue, leftValue};
}
} // namespace mlCore::autoDiff::binaryOperators
