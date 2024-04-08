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
						  { return (lhs->getOutputShape() != rhs->getOutputShape()); }) != inputs.cend())
	{
		LOG_ERROR("AutoDiff::Ops", "Expected input shapes to be equal!");
	}
}

OperatorPtr updateOp(const OperatorPtr& op)
{
	op->updateValue();
	return op;
}
} // namespace

OperatorPtr add(const NodePtr& lhsNode, const NodePtr& rhsNode)
{
	throwIfShapesUnequal({lhsNode, rhsNode});

	const auto fFunc = [](const std::vector<NodePtr>& inputs)
	{ return inputs.front()->getValue() + inputs.back()->getValue(); };

	const auto bFunc = [](const std::vector<NodePtr>&) {
		return std::vector{mlCore::Tensor{1.0}, mlCore::Tensor{1.0}};
	};

	return updateOp(std::make_shared<detail::PlainChainRuleOp>(std::vector{lhsNode, rhsNode}, fFunc, bFunc));
}

OperatorPtr subtract(const NodePtr& lhsNode, const NodePtr& rhsNode)
{
	throwIfShapesUnequal({lhsNode, rhsNode});

	const auto fFunc = [](const std::vector<NodePtr>& inputs)
	{ return inputs.front()->getValue() - inputs.back()->getValue(); };

	const auto bFunc = [](const std::vector<NodePtr>&) {
		return std::vector{mlCore::Tensor{1.0}, mlCore::Tensor{-1.0}};
	};

	return updateOp(std::make_shared<detail::PlainChainRuleOp>(std::vector{lhsNode, rhsNode}, fFunc, bFunc));
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

		derivatives.emplace_back(inputs.back()->getValue());
		derivatives.emplace_back(inputs.front()->getValue());

		return derivatives;
	};

	return updateOp(std::make_shared<detail::PlainChainRuleOp>(std::vector{lhsNode, rhsNode}, fFunc, bFunc));
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

		derivatives.emplace_back(std::move(leftDeriv));
		derivatives.emplace_back(std::move(rightDeriv));

		return derivatives;
	};

	return updateOp(std::make_shared<detail::PlainChainRuleOp>(std::vector{lhsNode, rhsNode}, fFunc, bFunc));
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

	return updateOp(std::make_shared<detail::MatMulOp>(std::vector{lhsNode, rhsNode}));
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

		derivatives.emplace_back(ones / inputs.front()->getValue());

		return derivatives;
	};

	return updateOp(std::make_shared<detail::PlainChainRuleOp>(std::vector{node}, fFunc, bFunc));
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

		derivatives.emplace_back(std::move(inputCopy));

		return derivatives;
	};

	return updateOp(std::make_shared<detail::PlainChainRuleOp>(std::vector{node}, fFunc, bFunc));
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
			val = 1.0 / (1.0 + std::pow(M_E, -val));
			val = val * (1.0 - val);
		}

		std::vector<mlCore::Tensor> derivatives;
		derivatives.reserve(1);

		derivatives.emplace_back(std::move(inputCopy));

		return derivatives;
	};

	return updateOp(std::make_shared<detail::PlainChainRuleOp>(std::vector{node}, fFunc, bFunc));
}
} // namespace autoDiff::ops
