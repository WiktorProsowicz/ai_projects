#include "AutoDiff/BinaryOperators/PowerOperator.h"

#include "MLCore/TensorOperations.h"

namespace mlCore::autoDiff::binaryOperators
{
void PowerOperator::updateValue()
{
	value_ = TensorOperations::power(lhsInput_->getValue(), rhsInput_->getValue());
}

std::pair<Tensor, Tensor> PowerOperator::computeDerivative(const Tensor& outerDerivative) const
{
	auto [leftDerivative, rightDerivative] = computeDirectDerivative();

	return {leftDerivative * outerDerivative, rightDerivative * outerDerivative};
}

std::pair<Tensor, Tensor> PowerOperator::computeDirectDerivative() const
{
	const auto& [leftInputNode, rightInputNode] = getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	return {TensorOperations::power(leftValue, rightValue - Tensor(rightValue.shape(), 1)) * rightValue,
			TensorOperations::ln(leftValue) * TensorOperations::power(leftValue, rightValue)};
}
} // namespace mlCore::autoDiff::binaryOperators
