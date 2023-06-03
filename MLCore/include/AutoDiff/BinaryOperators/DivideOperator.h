#ifndef BINARYOPERATORS_DIVIDEOPERATOR_H
#define BINARYOPERATORS_DIVIDEOPERATOR_H

#include <AutoDiff/BinaryOperators/IBinaryOperator.h>

namespace mlCore
{
class DivideOperator final : public IBinaryOperator
{
public:
	DivideOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: IBinaryOperator(lhsInput, rhsInput){};

	virtual void updateValue() override;
};

using DivideOperatorPtr = std::shared_ptr<DivideOperator>;

} // namespace mlCore

#endif