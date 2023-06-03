#ifndef UNARYOPERATORS_RELUOPERATOR_H
#define UNARYOPERATORS_RELUOPERATOR_H

#include <AutoDiff/UnaryOperators/IUnaryOperator.h>

namespace mlCore
{
/**
 * @brief ReluOperator represents REctified Linear Unit tensor operation
 * 
 */
class ReluOperator final : public IUnaryOperator
{
public:
	ReluOperator(const NodePtr input)
		: IUnaryOperator(input){};

	virtual void updateValue() override;
};

using ReluOperatorPtr = std::shared_ptr<ReluOperator>;

} // namespace mlCore

#endif