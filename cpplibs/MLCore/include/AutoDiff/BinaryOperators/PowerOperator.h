#ifndef BINARYOPERATORS_POWEROPERATOR_H
#define BINARYOPERATORS_POWEROPERATOR_H

#include <AutoDiff/BinaryOperators/BinaryOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
class PowerOperator final : public BinaryOperator
{
public:
	PowerOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: BinaryOperator(lhsInput, rhsInput){};

	void updateValue() override;

	std::pair<Tensor, Tensor> computeDerivative(const Tensor& outerDerivative) const override;

	std::pair<Tensor, Tensor> computeDirectDerivative() const override;
};

using PowerOperatorPtr = std::shared_ptr<PowerOperator>;

} // namespace mlCore::autoDiff::binaryOperators

#endif
