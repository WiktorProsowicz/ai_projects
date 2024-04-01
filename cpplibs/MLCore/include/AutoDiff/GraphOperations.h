#ifndef AUTODIFF_GRAPHOPERATIONS_H
#define AUTODIFF_GRAPHOPERATIONS_H

#include "AutoDiff/ComputationGraph.h"
#include "AutoDiff/GraphNodes.hpp"

/**
 * @brief Algorithms operating on GraphNodes.
 */
namespace mlCore::autoDiff
{

/// Concept for functions taking any number of shared pointers of types inheriting from Node and returning
/// analogous type
template <typename Operation, typename... NodePtrs>
concept NodeOperation = requires(NodePtrs... inputNodes, Operation oper) {
	// input types are shared pointers
	(... && std::is_same_v<std::shared_ptr<decltype(*inputNodes)>, decltype(inputNodes)>);

	// pointed types are derived from Node
	(... && std::is_base_of_v<Node, decltype(*inputNodes)>);

	// result type is shared ptr
	std::is_same_v<std::shared_ptr<decltype(*oper(inputNodes...))>, decltype(oper(inputNodes...))>;

	// result pointer points to something derived from Node
	std::is_base_of_v<Node, decltype(*(oper(inputNodes...)))>;
};

/**
 * @brief Performs given operation on input nodes returning NodePtr and adds the result to ComputationGraph if
 * provided.
 *
 * @param operation Operation complying with NodeOperation concept.
 * @param graph Pointer to computation graph instance to which the result will be added.
 * @param inputNodes Arguments for `operation`.
 * @return Result of the `operation` based on `inputNodes`.
 */
template <typename Operation, typename... NodePtrs>
NodePtr performAndAdd(Operation operation,
					  const std::shared_ptr<ComputationGraph>& graph,
					  NodePtrs... inputNodes) requires NodeOperation<Operation, NodePtrs...>
{
	auto result = operation(inputNodes...);

	if(graph && graph->isActive())
	{
		graph->addNode(result);
	}

	return result;
}

namespace binaryOperations
{

NodePtr multiply(const NodePtr& lNode, const NodePtr& rNode);
NodePtr add(const NodePtr& lNode, const NodePtr& rNode);
NodePtr subtract(const NodePtr& lNode, const NodePtr& rNode);
NodePtr divide(const NodePtr& lNode, const NodePtr& rNode);
NodePtr matmul(const NodePtr& lNode, const NodePtr& rNode);
NodePtr power(const NodePtr& baseNode, const NodePtr& factorNode);

} // namespace binaryOperations

namespace unaryOperations
{
NodePtr ln(const NodePtr& node);
} // namespace unaryOperations

namespace nodesActivations
{

NodePtr relu(const NodePtr& node);
NodePtr sigmoid(const NodePtr& node);

} // namespace nodesActivations
} // namespace mlCore::autoDiff

#endif
