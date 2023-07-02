#include <AutoDiff/BinaryOperators/AddOperator.h>
#include <AutoDiff/BinaryOperators/DivideOperator.h>
#include <AutoDiff/BinaryOperators/MatmulOperator.h>
#include <AutoDiff/BinaryOperators/MultiplyOperator.h>
#include <AutoDiff/BinaryOperators/PowerOperator.h>
#include <AutoDiff/BinaryOperators/SubtractOperator.h>

#include <AutoDiff/GraphOperations.h>
#include <AutoDiff/UnaryOperators/ReluOperator.h>
#include <AutoDiff/UnaryOperators/SigmoidOperator.h>

namespace mlCore
{
/****************
 * 
 * Binary operators
 * 
 ****************/

BinaryOperations::BinaryOperations(const std::shared_ptr<ComputationGraph> graph)
	: graph_(graph)
{ }

BinaryOperations::BinaryOperations()
	: graph_(nullptr)
{ }

template <typename ResultNodeType>
std::shared_ptr<ResultNodeType> BinaryOperations::operationImpl(const NodePtr lNode,
																const NodePtr rNode)
{
	auto resultNode = std::make_shared<ResultNodeType>(lNode, rNode);

	resultNode->updateValue();

	if(graph_ && graph_->isActive())
		graph_->addNode(resultNode);

	return resultNode;
}

NodePtr BinaryOperations::multiply(const NodePtr lNode, const NodePtr rNode)
{
	return BinaryOperations{}.operationImpl<MultiplyOperator>(lNode, rNode);
}

NodePtr BinaryOperations::add(const NodePtr lNode, const NodePtr rNode)
{
	return BinaryOperations{}.operationImpl<AddOperator>(lNode, rNode);
}

NodePtr BinaryOperations::subtract(const NodePtr lNode, const NodePtr rNode)
{
	return BinaryOperations{}.operationImpl<SubtractOperator>(lNode, rNode);
}

NodePtr BinaryOperations::divide(const NodePtr lNode, const NodePtr rNode)
{
	return BinaryOperations{}.operationImpl<DivideOperator>(lNode, rNode);
}

NodePtr BinaryOperations::power(const NodePtr lNode, const NodePtr rNode)
{
	return BinaryOperations{}.operationImpl<PowerOperator>(lNode, rNode);
}

NodePtr BinaryOperations::matmul(const NodePtr lNode, const NodePtr rNode)
{
	return BinaryOperations{}.operationImpl<MatmulOperator>(lNode, rNode);
}

/****************
 * 
 * Unary operators
 * 
 ****************/

/****************
 * 
 * Activations
 * 
 ****************/

NodesActivations::NodesActivations(const std::shared_ptr<ComputationGraph> graph)
	: graph_(graph)
{ }

NodesActivations::NodesActivations()
	: graph_(nullptr)
{ }

template <typename ResultNodeType>
std::shared_ptr<ResultNodeType> NodesActivations::operationImpl(const NodePtr node)
{
	auto resultNode = std::make_shared<ResultNodeType>(node);

	resultNode->updateValue();

	if(graph_ && graph_->isActive())
	{
		graph_->addNode(resultNode);
	}

	return resultNode;
}

NodePtr NodesActivations::relu(const NodePtr node)
{
	return operationImpl<ReluOperator>(node);
}

NodePtr NodesActivations::sigmoid(const NodePtr node)
{
	return operationImpl<SigmoidOperator>(node);
}
} // namespace mlCore