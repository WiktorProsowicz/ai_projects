#include <AutoDiff/BinaryOperators/AddOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
void AddOperator::updateValue()
{
	value_ = lhsInput_->getValue() + rhsInput_->getValue();
}

std::pair<Tensor, Tensor> AddOperator::computeDerivative(const Tensor& outerDerivative) const
{
	return {outerDerivative, outerDerivative};
}

std::pair<Tensor, Tensor> AddOperator::computeDirectDerivative() const
{
	const auto& [leftInput, rightInput] = getInputs();

	return {Tensor(leftInput->getValue().shape(), 1.0), Tensor(rightInput->getValue().shape(), 1.0)};
}
} // namespace mlCore::autoDiff::binaryOperators
