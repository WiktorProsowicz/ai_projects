#ifndef BINARYOPERATORS_SUBTRACTOPERATOR_H
#define BINARYOPERATORS_SUBTRACTOPERATOR_H

#include <AutoDiff/BinaryOperators/BinaryOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
class SubtractOperator final : public BinaryOperator
{
public:
	SubtractOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: BinaryOperator(lhsInput, rhsInput){};

	void updateValue() override;

	std::pair<Tensor, Tensor> computeDerivative(const Tensor& outerDerivative) const override;

	std::pair<Tensor, Tensor> computeDirectDerivative() const override;
};

using SubtractOperatorPtr = std::shared_ptr<SubtractOperator>;

} // namespace mlCore::autoDiff::binaryOperators

#endif
