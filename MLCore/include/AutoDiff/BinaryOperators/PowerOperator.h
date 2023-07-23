#ifndef BINARYOPERATORS_POWEROPERATOR_H
#define BINARYOPERATORS_POWEROPERATOR_H

#include <AutoDiff/BinaryOperators/IBinaryOperator.h>

namespace mlCore::autoDiff::binaryOperators
{
class PowerOperator final : public IBinaryOperator
{
public:
	PowerOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: IBinaryOperator(lhsInput, rhsInput){};

	void updateValue() override;
};

using PowerOperatorPtr = std::shared_ptr<PowerOperator>;

} // namespace mlCore::autoDiff::binaryOperators

#endif