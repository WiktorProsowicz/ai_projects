#include "AutoDiff/Operations.h"

#include <memory>

#include "AutoDiff/Operators/MatMulOp.hpp"
#include "AutoDiff/Operators/PlainChainRuleOp.hpp"
#include "MLCore/TensorOperations.h"

namespace autoDiff::ops
{
namespace
{
/**
 * @brief Checks if output shapes of the given input nodes are equal.
 *
 * @throws std::runtime_error If the shapes are not equal.
 */
void throwIfShapesUnequal(const std::vector<NodePtr>& inputs)
{

	if(std::adjacent_find(inputs.cbegin(),
						  inputs.cend(),
						  [](const NodePtr& lhs, const NodePtr& rhs)
						  { return (lhs->getOutputShape() != rhs->getOutputShape()); }) == inputs.cend())
	{
		LOG_ERROR("AutoDiff::Ops", "Expected input shapes to be equal!");
	}
}
} // namespace

OperatorPtr add(const NodePtr& lhsNode, const NodePtr& rhsNode)
{
	throwIfShapesUnequal({lhsNode, rhsNode});

	const auto fFunc = [](const std::vector<NodePtr>& inputs)
	{ return inputs.front()->getValue() + inputs.back()->getValue(); };

	const auto bFunc = [](const std::vector<NodePtr>& inputs) {
		return std::vector{mlCore::Tensor{1.0}, mlCore::Tensor{1.0}};
	};

	return std::make_shared<detail::PlainChainRuleOp>(std::vector{lhsNode, rhsNode}, fFunc, bFunc);
}

OperatorPtr subtract(const NodePtr& lhsNode, const NodePtr& rhsNode)
{
	throwIfShapesUnequal({lhsNode, rhsNode});

	const auto fFunc = [](const std::vector<NodePtr>& inputs)
	{ return inputs.front()->getValue() - inputs.back()->getValue(); };

	const auto bFunc = [](const std::vector<NodePtr>& inputs) {
		return std::vector{mlCore::Tensor{1.0}, mlCore::Tensor{1.0}};
	};

	return std::make_shared<detail::PlainChainRuleOp>(std::vector{lhsNode, rhsNode}, fFunc, bFunc);
}

OperatorPtr multiply(const NodePtr& lhsNode, const NodePtr& rhsNode)
{
	throwIfShapesUnequal({lhsNode, rhsNode});

	const auto fFunc = [](const std::vector<NodePtr>& inputs)
	{ return inputs.front()->getValue() * inputs.back()->getValue(); };

	const auto bFunc = [](const std::vector<NodePtr>& inputs)
	{
		std::vector<mlCore::Tensor> derivatives;
		derivatives.reserve(2);

		derivatives[0] = inputs.back()->getValue();
		derivatives[1] = inputs.front()->getValue();

		return derivatives;
	};

	return std::make_shared<detail::PlainChainRuleOp>(std::vector{lhsNode, rhsNode}, fFunc, bFunc);
}

OperatorPtr divide(const NodePtr& lhsNode, const NodePtr& rhsNode)
{
	throwIfShapesUnequal({lhsNode, rhsNode});

	const auto fFunc = [](const std::vector<NodePtr>& inputs)
	{ return inputs.front()->getValue() / inputs.back()->getValue(); };

	const auto bFunc = [](const std::vector<NodePtr>& inputs)
	{
		std::vector<mlCore::Tensor> derivatives;
		derivatives.reserve(2);

		auto leftDeriv = mlCore::Tensor(inputs.back()->getOutputShape(), 1.0) / inputs.back()->getValue();
		auto rightDeriv =
			-inputs.front()->getValue() / (inputs.back()->getValue() * inputs.back()->getValue());

		derivatives[0] = std::move(leftDeriv);
		derivatives[1] = std::move(rightDeriv);

		return derivatives;
	};

	return std::make_shared<detail::PlainChainRuleOp>(std::vector{lhsNode, rhsNode}, fFunc, bFunc);
}

OperatorPtr matmul(const NodePtr& lhsNode, const NodePtr& rhsNode)
{
	try
	{
		mlCore::detail::getOutputShapeForMatmul(lhsNode->getOutputShape(), rhsNode->getOutputShape());
	}
	catch(const std::runtime_error& error)
	{
		LOG_ERROR("AutoDiff::Ops", error.what());
	}

	return std::make_shared<detail::MatMulOp>(std::vector{lhsNode, rhsNode});
}

OperatorPtr naturalLog(const NodePtr& node)
{
	const auto fFunc = [](const std::vector<NodePtr>& inputs)
	{ return mlCore::TensorOperations::ln(inputs.front()->getValue()); };

	const auto bFunc = [](const std::vector<NodePtr>& inputs)
	{
		const mlCore::Tensor ones(inputs.front()->getOutputShape(), 1.0);

		std::vector<mlCore::Tensor> derivatives;
		derivatives.reserve(1);

		derivatives[0] = ones / inputs.front()->getValue();

		return derivatives;
	};

	return std::make_shared<detail::PlainChainRuleOp>(std::vector{node}, fFunc, bFunc);
}

OperatorPtr relu(const NodePtr& node)
{
	const auto fFunc = [](const std::vector<NodePtr>& inputs)
	{ return mlCore::TensorOperations::relu(inputs.front()->getValue()); };

	const auto bFunc = [](const std::vector<NodePtr>& inputs)
	{
		mlCore::Tensor inputCopy = inputs.front()->getValue();

		for(auto& val : inputCopy)
		{
			val = val > 0.0 ? 1.0 : 0.0;
		}

		std::vector<mlCore::Tensor> derivatives;
		derivatives.reserve(1);

		derivatives[0] = std::move(inputCopy);

		return derivatives;
	};

	return std::make_shared<detail::PlainChainRuleOp>(std::vector{node}, fFunc, bFunc);
}

OperatorPtr sigmoid(const NodePtr& node)
{
	const auto fFunc = [](const std::vector<NodePtr>& inputs)
	{ return mlCore::TensorOperations::sigmoid(inputs.front()->getValue()); };

	const auto bFunc = [](const std::vector<NodePtr>& inputs)
	{
		mlCore::Tensor inputCopy = inputs.front()->getValue();

		for(auto& val : inputCopy)
		{
			val = val * (1.0 - val);
		}

		std::vector<mlCore::Tensor> derivatives;
		derivatives.reserve(1);

		derivatives[0] = std::move(inputCopy);

		return derivatives;
	};

	return std::make_shared<detail::PlainChainRuleOp>(std::vector{node}, fFunc, bFunc);
}
} // namespace autoDiff::ops