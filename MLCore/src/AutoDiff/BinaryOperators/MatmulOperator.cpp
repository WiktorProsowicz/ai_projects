#include <AutoDiff/BinaryOperators/MatmulOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
void MatmulOperator::updateValue()
{
	value_ = lhsInput_->getValue().matmul(rhsInput_->getValue());
}

std::pair<Tensor, Tensor> MatmulOperator::computeDerivative(const Tensor& outerDerivative) const
{
	const auto& [leftInputNode, rightInputNode] = getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	return {outerDerivative.matmul(rightValue.transposed()),
			leftValue.transposed().matmul(outerDerivative)};
}

std::pair<Tensor, Tensor> MatmulOperator::computeDirectDerivative() const
{
	const auto& [leftInputNode, rightInputNode] = getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	const Tensor onesWithOutputShape(value_.shape(), 1.0);

	return {onesWithOutputShape.matmul(rightValue.transposed()),
			leftValue.transposed().matmul(onesWithOutputShape)};
}
} // namespace mlCore::autoDiff::binaryOperators