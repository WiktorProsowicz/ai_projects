#ifndef UNARYOPERATORS_SIGMOIDOPERATOR_H
#define UNARYOPERATORS_SIGMOIDOPERATOR_H

#include <AutoDiff/UnaryOperators/UnaryOperator.h>

namespace mlCore::autoDiff::unaryOperators
{
/**
 * @brief SigmoidOperator wraps sigmoid - a tensor unary operation
 * 
 */
class SigmoidOperator final : public UnaryOperator
{
public:
	SigmoidOperator(const NodePtr input)
		: UnaryOperator(input){};

	void updateValue() override;
};

using SigmoidOperatorPtr = std::shared_ptr<SigmoidOperator>;

} // namespace mlCore::autoDiff::unaryOperators

#endif