#ifndef BINARYOPERATORS_MULTIPLYOPERATOR_H
#define BINARYOPERATORS_MULTIPLYOPERATOR_H

#include <AutoDiff/BinaryOperators/IBinaryOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
class MultiplyOperator final : public IBinaryOperator
{
public:
	MultiplyOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: IBinaryOperator(lhsInput, rhsInput){};

	void updateValue() override;
};

using MultiplyOperatorPtr = std::shared_ptr<MultiplyOperator>;

} // namespace mlCore::autoDiff::binaryOperators

#endif