#ifndef MLCORE_IBINARYOPERATOR_H
#define MLCORE_IBINARYOPERATOR_H

#include <AutoDiff/GraphNodes.hpp>

namespace mlCore
{
/**
 * @brief Represents a result of operation on two nodes. Its internal processing depends on its type, wraps TensorFunctions algorithms
 * 
 */
class IBinaryOperator : public Node
{
public:
	IBinaryOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: Node(std::vector<size_t>{})
		, lhsInput_(lhsInput)
		, rhsInput_(rhsInput){};

	// tells the operator to compute its value based on its inputs
	virtual void updateValue() = 0;

	std::pair<NodePtr, NodePtr> getInputs() const
	{
		return {lhsInput_, rhsInput_};
	}

protected:
	const NodePtr lhsInput_;
	const NodePtr rhsInput_;
};

using IBinaryOperatorPtr = std::shared_ptr<IBinaryOperator>;

} // namespace mlCore

#endif