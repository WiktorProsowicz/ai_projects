#include "AutoDiff/BinaryOperators/AddOperator.h"

#include <utility>

namespace mlCore::autoDiff::binaryOperators
{
void AddOperator::updateValue()
{
	_value = _lhsInput->getValue() + _rhsInput->getValue();
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
