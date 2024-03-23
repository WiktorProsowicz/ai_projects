#ifndef UNARYOPERATORS_RELUOPERATOR_H
#define UNARYOPERATORS_RELUOPERATOR_H

#include <AutoDiff/UnaryOperators/UnaryOperator.h>

namespace mlCore::autoDiff::unaryOperators
{
/**
 * @brief ReluOperator represents REctified Linear Unit tensor operation
 *
 */
class ReluOperator final : public UnaryOperator
{
public:
	ReluOperator(const NodePtr input)
		: UnaryOperator(input){};

	void updateValue() override;

	Tensor computeDerivative(const Tensor& outerDerivative) const override;

	Tensor computeDirectDerivative() const override;
};

using ReluOperatorPtr = std::shared_ptr<ReluOperator>;

} // namespace mlCore::autoDiff::unaryOperators

#endif
