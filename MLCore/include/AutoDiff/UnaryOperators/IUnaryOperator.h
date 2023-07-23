#ifndef UNARYOPERATORS_IUNARYOPERATOR_H
#define UNARYOPERATORS_IUNARYOPERATOR_H

#include <AutoDiff/GraphNodes.hpp>

namespace mlCore::autoDiff::unaryOperators
{
/**
 * @brief Represents a result of modification of a single node. Its internal processing depends on its subclass, wraps TensorFunctions algorithms
 * 
 */
class IUnaryOperator : public Node
{
public:
	IUnaryOperator(const NodePtr input)
		: Node(std::vector<size_t>{})
		, input_(input){};

	// tells the operator to compute its value based om its input
	virtual void updateValue() = 0;

	NodePtr getInput() const
	{
		return input_;
	}

protected:
	NodePtr input_;
};

using IUnaryOperatorPtr = std::shared_ptr<IUnaryOperator>;

} // namespace mlCore::autoDiff::unaryOperators

#endif