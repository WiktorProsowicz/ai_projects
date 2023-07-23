#ifndef BINARYOPERATORS_MATMULOPERATOR_H
#define BINARYOPERATORS_MATMULOPERATOR_H

#include <AutoDiff/BinaryOperators/BinaryOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
class MatmulOperator final : public BinaryOperator
{
public:
	MatmulOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: BinaryOperator(lhsInput, rhsInput){};

	void updateValue() override;
};

using MatmulOperatorPtr = std::shared_ptr<MatmulOperator>;

} // namespace mlCore::autoDiff::binaryOperators

#endif