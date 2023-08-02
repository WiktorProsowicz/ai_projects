#ifndef BINARYOPERATORS_MULTIPLYOPERATOR_H
#define BINARYOPERATORS_MULTIPLYOPERATOR_H

#include <AutoDiff/BinaryOperators/BinaryOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
class MultiplyOperator final : public BinaryOperator
{
public:
	MultiplyOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: BinaryOperator(lhsInput, rhsInput){};

	void updateValue() override;

	std::pair<Tensor, Tensor> computeDerivative(const Tensor& outerDerivative) const override;

	std::pair<Tensor, Tensor> computeDirectDerivative() const override;
};

using MultiplyOperatorPtr = std::shared_ptr<MultiplyOperator>;

} // namespace mlCore::autoDiff::binaryOperators

#endif