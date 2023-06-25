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

	NodePtr multiply(const NodePtr lNode, const NodePtr rNode);
	NodePtr add(const NodePtr lNode, const NodePtr rNode);
	NodePtr subtract(const NodePtr lNode, const NodePtr rNode);
	NodePtr divide(const NodePtr lNode, const NodePtr rNode);
	NodePtr matmul(const NodePtr lNode, const NodePtr rNode);
	NodePtr power(const NodePtr baseNode, const NodePtr factorNode);

private:
	template <typename ResultNodeType>
	std::shared_ptr<ResultNodeType> operationImpl(const NodePtr lNode, const NodePtr rNode);

private:
	std::shared_ptr<ComputationGraph> graph_;
};

class UnaryOperations
{ };

class NodesActivations
{
public:
	NodesActivations();
	NodesActivations(const std::shared_ptr<ComputationGraph> graph);

	NodesActivations(const NodesActivations&) = delete; // Copy ctor
	NodesActivations(NodesActivations&&) = delete; // Move ctor
	NodesActivations& operator=(const NodesActivations&) = delete; // Copy assignment
	NodesActivations& operator=(NodesActivations&&) = delete; // Move assignment

	NodePtr relu(const NodePtr node);
	NodePtr sigmoid(const NodePtr node);

private:
	template <typename ResultNodeType>
	std::shared_ptr<ResultNodeType> operationImpl(const NodePtr lNode);

private:
	std::shared_ptr<ComputationGraph> graph_;
};

} // namespace mlCore

#endif