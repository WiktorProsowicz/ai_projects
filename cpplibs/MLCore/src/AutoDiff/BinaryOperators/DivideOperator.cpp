#include "AutoDiff/BinaryOperators/DivideOperator.h"

namespace mlCore::autoDiff::binaryOperators
{
void DivideOperator::updateValue()
{
	_value = _lhsInput->getValue() / _rhsInput->getValue();
}

std::pair<Tensor, Tensor> DivideOperator::computeDerivative(const Tensor& outerDerivative) const
{
	auto [leftDerivative, rightDerivative] = computeDirectDerivative();

	return {leftDerivative * outerDerivative, rightDerivative * outerDerivative};
}

std::pair<Tensor, Tensor> DivideOperator::computeDirectDerivative() const
{
	const auto& [leftInputNode, rightInputNode] = getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	return {(Tensor(rightValue.shape(), 1.0) / rightValue), -leftValue / (rightValue * rightValue)};
}
} // namespace mlCore::autoDiff::binaryOperators
