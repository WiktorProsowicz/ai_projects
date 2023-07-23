#ifndef BINARYOPERATORS_DIVIDEOPERATOR_H
#define BINARYOPERATORS_DIVIDEOPERATOR_H

#include <AutoDiff/BinaryOperators/BinaryOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
class DivideOperator final : public BinaryOperator
{
public:
	DivideOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: BinaryOperator(lhsInput, rhsInput){};

	void updateValue() override;
};

using DivideOperatorPtr = std::shared_ptr<DivideOperator>;

} // namespace mlCore::autoDiff::binaryOperators

#endif