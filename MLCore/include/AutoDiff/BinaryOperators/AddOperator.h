#ifndef BINARYOPERATORS_ADDOPERATOR_H
#define BINARYOPERATORS_ADDOPERATOR_H

#include <AutoDiff/BinaryOperators/IBinaryOperator.h>

namespace mlCore
{
class AddOperator final : public IBinaryOperator
{
public:
	AddOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: IBinaryOperator(lhsInput, rhsInput){};

	virtual void updateValue() override;
};

using AddOperatorPtr = std::shared_ptr<AddOperator>;

} // namespace mlCore

#endif