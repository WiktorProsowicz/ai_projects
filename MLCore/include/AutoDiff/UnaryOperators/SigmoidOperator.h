#ifndef UNARYOPERATORS_SIGMOIDOPERATOR_H
#define UNARYOPERATORS_SIGMOIDOPERATOR_H

#include <AutoDiff/UnaryOperators/IUnaryOperator.h>

namespace mlCore
{
/**
 * @brief SigmoidOperator wraps sigmoid - a tensor unary operation
 * 
 */
class SigmoidOperator final : public IUnaryOperator
{
public:
	SigmoidOperator(const NodePtr input)
		: IUnaryOperator(input){};

	virtual void updateValue() override;
};

using SigmoidOperatorPtr = std::shared_ptr<SigmoidOperator>;

} // namespace mlCore

#endif