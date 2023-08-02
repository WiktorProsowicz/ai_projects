#include <AutoDiff/GraphOperations.h>

#include <AutoDiff/BinaryOperators/AddOperator.h>
#include <AutoDiff/BinaryOperators/DivideOperator.h>
#include <AutoDiff/BinaryOperators/MatmulOperator.h>
#include <AutoDiff/BinaryOperators/MultiplyOperator.h>
#include <AutoDiff/BinaryOperators/PowerOperator.h>
#include <AutoDiff/BinaryOperators/SubtractOperator.h>

#include <AutoDiff/UnaryOperators/LnOperator.h>
#include <AutoDiff/UnaryOperators/ReluOperator.h>
#include <AutoDiff/UnaryOperators/SigmoidOperator.h>

namespace mlCore::autoDiff
{

/****************
 * 
 * Binary
 * 
 ****************/
namespace binaryOperations
{
NodePtr multiply(const NodePtr lNode, const NodePtr rNode)
{
	return std::make_shared<binaryOperators::MultiplyOperator>(lNode, rNode);
}

NodePtr add(const NodePtr lNode, const NodePtr rNode)
{
	return std::make_shared<binaryOperators::AddOperator>(lNode, rNode);
}

NodePtr subtract(const NodePtr lNode, const NodePtr rNode)
{
	return std::make_shared<binaryOperators::SubtractOperator>(lNode, rNode);
}

NodePtr divide(const NodePtr lNode, const NodePtr rNode)
{
	return std::make_shared<binaryOperators::DivideOperator>(lNode, rNode);
}

NodePtr power(const NodePtr baseNode, const NodePtr factorNode)
{
	return std::make_shared<binaryOperators::PowerOperator>(baseNode, factorNode);
}

NodePtr matmul(const NodePtr lNode, const NodePtr rNode)
{
	return std::make_shared<binaryOperators::MatmulOperator>(lNode, rNode);
}
} // namespace binaryOperations

/****************
 * 
 * Activations
 * 
 ****************/

namespace unaryOperations
{
NodePtr ln(NodePtr node)
{
	return std::make_shared<unaryOperators::LnOperator>(node);
}
} // namespace unaryOperations

/****************
 * 
 * Activations
 * 
 ****************/

namespace nodesActivations
{
NodePtr relu(const NodePtr node)
{
	return std::make_shared<unaryOperators::ReluOperator>(node);
}

NodePtr sigmoid(const NodePtr node)
{
	return std::make_shared<unaryOperators::SigmoidOperator>(node);
}
} // namespace nodesActivations
} // namespace mlCore::autoDiff