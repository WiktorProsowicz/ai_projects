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
namespace
{
/// Creates an binary operator node of the provided type and updates its value.
template <typename BinaryOperator>
NodePtr binaryOperationImpl(const NodePtr& lNode, const NodePtr& rNode)
{
	auto result = std::make_shared<BinaryOperator>(lNode, rNode);

	result->updateValue();

	return result;
}
} // namespace
namespace binaryOperations
{
NodePtr multiply(const NodePtr lNode, const NodePtr rNode)
{
	return binaryOperationImpl<binaryOperators::MultiplyOperator>(lNode, rNode);
}

NodePtr add(const NodePtr lNode, const NodePtr rNode)
{
	return binaryOperationImpl<binaryOperators::AddOperator>(lNode, rNode);
}

NodePtr subtract(const NodePtr lNode, const NodePtr rNode)
{
	return binaryOperationImpl<binaryOperators::SubtractOperator>(lNode, rNode);
}

NodePtr divide(const NodePtr lNode, const NodePtr rNode)
{
	return binaryOperationImpl<binaryOperators::DivideOperator>(lNode, rNode);
}

NodePtr power(const NodePtr baseNode, const NodePtr factorNode)
{
	return binaryOperationImpl<binaryOperators::PowerOperator>(baseNode, factorNode);
}

NodePtr matmul(const NodePtr lNode, const NodePtr rNode)
{
	return binaryOperationImpl<binaryOperators::MatmulOperator>(lNode, rNode);
}
} // namespace binaryOperations

namespace
{
/// Creates an unary operator node of the provided type and updates its value.
template <typename UnaryOperator>
NodePtr unaryOperationImpl(const NodePtr& node)
{
	auto result = std::make_shared<UnaryOperator>(node);

	result->updateValue();

	return result;
}
} // namespace

/****************
 * 
 * Activations
 * 
 ****************/

namespace unaryOperations
{
NodePtr ln(NodePtr node)
{
	return unaryOperationImpl<unaryOperators::LnOperator>(node);
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
	return unaryOperationImpl<unaryOperators::ReluOperator>(node);
}

NodePtr sigmoid(const NodePtr node)
{
	return unaryOperationImpl<unaryOperators::SigmoidOperator>(node);
}
} // namespace nodesActivations
} // namespace mlCore::autoDiff