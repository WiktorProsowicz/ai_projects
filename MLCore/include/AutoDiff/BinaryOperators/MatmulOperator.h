#ifndef BINARYOPERATORS_MATMULOPERATOR_H
#define BINARYOPERATORS_MATMULOPERATOR_H

#include <AutoDiff/BinaryOperators/IBinaryOperator.h>

namespace mlCore
{
class MatmulOperator final : public IBinaryOperator
{
public:
	MatmulOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: IBinaryOperator(lhsInput, rhsInput){};

	void updateValue() override;
};

using MatmulOperatorPtr = std::shared_ptr<MatmulOperator>;

} // namespace mlCore

#endif