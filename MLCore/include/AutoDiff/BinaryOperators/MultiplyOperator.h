#ifndef BINARYOPERATORS_MULTIPLYOPERATOR_H
#define BINARYOPERATORS_MULTIPLYOPERATOR_H

#include <AutoDiff/BinaryOperators/IBinaryOperator.h>

namespace mlCore
{
class MultiplyOperator final : public IBinaryOperator
{
public:
	MultiplyOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: IBinaryOperator(lhsInput, rhsInput){};

	virtual void updateValue() override;
};

using MultiplyOperatorPtr = std::shared_ptr<MultiplyOperator>;

} // namespace mlCore

#endif