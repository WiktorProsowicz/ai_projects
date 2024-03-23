#ifndef BINARYOPERATORS_ADDOPERATOR_H
#define BINARYOPERATORS_ADDOPERATOR_H

#include <AutoDiff/BinaryOperators/BinaryOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
class AddOperator final : public BinaryOperator
{
public:
	AddOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: BinaryOperator(lhsInput, rhsInput){};

	void updateValue() override;

	std::pair<Tensor, Tensor> computeDerivative(const Tensor& outerDerivative) const override;

	std::pair<Tensor, Tensor> computeDirectDerivative() const override;
};

using AddOperatorPtr = std::shared_ptr<AddOperator>;

} // namespace mlCore::autoDiff::binaryOperators

#endif
