#ifndef BINARYOPERATORS_SUBTRACTOPERATOR_H
#define BINARYOPERATORS_SUBTRACTOPERATOR_H

#include <AutoDiff/BinaryOperators/IBinaryOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
class SubtractOperator final : public IBinaryOperator
{
public:
	SubtractOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: IBinaryOperator(lhsInput, rhsInput){};

	void updateValue() override;
};

using SubtractOperatorPtr = std::shared_ptr<SubtractOperator>;

} // namespace mlCore::autoDiff::binaryOperators

#endif