#ifndef MLCORE_INCLUDE_AUTODIFF_UNARYOPERATORS_LNOPERATOR_H
#define MLCORE_INCLUDE_AUTODIFF_UNARYOPERATORS_LNOPERATOR_H

#include "AutoDiff/UnaryOperators/UnaryOperator.h"

namespace mlCore::autoDiff::unaryOperators
{
/**
 * @brief LnOperator represents natural logarithm operation performed on contained value.
 *
 */
class LnOperator final : public UnaryOperator
{
public:
	explicit LnOperator(const NodePtr& input)
		: UnaryOperator(input){};

	void updateValue() override;

	Tensor computeDerivative(const Tensor& outerDerivative) const override;

	Tensor computeDirectDerivative() const override;
};

using LnOperatorPtr = std::shared_ptr<LnOperator>;

} // namespace mlCore::autoDiff::unaryOperators

#endif
