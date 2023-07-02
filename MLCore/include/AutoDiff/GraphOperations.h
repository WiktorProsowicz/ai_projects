#ifndef AUTODIFF_GRAPHOPERATIONS_H
#define AUTODIFF_GRAPHOPERATIONS_H

#include <AutoDiff/ComputationGraph.h>
#include <AutoDiff/GraphNodes.h>

/**
 * @brief Algorithms operating on GraphNodes, cooperating with ComputationGraph
*/
namespace mlCore
{
class BinaryOperations
{
public:
	BinaryOperations();
	BinaryOperations(const std::shared_ptr<ComputationGraph> graph);

	BinaryOperations(const BinaryOperations&) = delete; // Copy ctor
	BinaryOperations(BinaryOperations&&) = delete; // Move ctor
	BinaryOperations& operator=(const BinaryOperations&) = delete; // Copy assignment
	BinaryOperations& operator=(BinaryOperations&&) = delete; // Move assignment

	NodePtr multiply(NodePtr lNode, NodePtr rNode);
	NodePtr add(NodePtr lNode, NodePtr rNode);
	NodePtr subtract(NodePtr lNode, NodePtr rNode);
	NodePtr divide(NodePtr lNode, NodePtr rNode);
	NodePtr matmul(NodePtr lNode, NodePtr rNode);
	NodePtr power(NodePtr baseNode, NodePtr factorNode);

private:
	template <typename ResultNodeType>
	std::shared_ptr<ResultNodeType> operationImpl(NodePtr lNode, NodePtr rNode);

private:
	std::shared_ptr<ComputationGraph> graph_;
};

class UnaryOperations
{ };

class NodesActivations
{
public:
	NodesActivations();
	NodesActivations(std::shared_ptr<ComputationGraph> graph);

	NodesActivations(const NodesActivations&) = delete; // Copy ctor
	NodesActivations(NodesActivations&&) = delete; // Move ctor
	NodesActivations& operator=(const NodesActivations&) = delete; // Copy assignment
	NodesActivations& operator=(NodesActivations&&) = delete; // Move assignment

	NodePtr relu(NodePtr node);
	NodePtr sigmoid(NodePtr node);

private:
	template <typename ResultNodeType>
	std::shared_ptr<ResultNodeType> operationImpl(NodePtr lNode);

private:
	std::shared_ptr<ComputationGraph> graph_;
};

} // namespace mlCore

#endif