#include "AutoDiff/BinaryOperators/SubtractOperator.h"

#include <utility>

namespace mlCore::autoDiff::binaryOperators
{
void SubtractOperator::updateValue()
{
	_value = _lhsInput->getValue() - _rhsInput->getValue();
}

std::pair<Tensor, Tensor> SubtractOperator::computeDerivative(const Tensor& outerDerivative) const
{
	return {outerDerivative, -outerDerivative};
}

std::pair<Tensor, Tensor> SubtractOperator::computeDirectDerivative() const
{
	const auto& [lhs, rhs] = getInputs();

	return {Tensor(lhs->getValue().shape(), 1.0), Tensor(rhs->getValue().shape(), -1.0)};
}
} // namespace mlCore::autoDiff::binaryOperators
