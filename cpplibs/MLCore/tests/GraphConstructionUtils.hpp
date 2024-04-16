#ifndef MLCORE_TESTS_GRAPHCONSTRUCTIONUTILS_HPP
#define MLCORE_TESTS_GRAPHCONSTRUCTIONUTILS_HPP

#include "OperatorStatsUtils.hpp"

namespace mlCoreTests
{
/// Decorates input `node` or returns it if it is not of operator type.
autoDiff::NodePtr wrapNode(const std::vector<autoDiff::NodePtr>& inputs,
						   const autoDiff::NodePtr& node,
						   const std::shared_ptr<OperatorStats>& stats)
{
	using namespace autoDiff;

	if(const auto casted = std::dynamic_pointer_cast<Operator>(node))
	{
		return std::make_shared<OperatorDecorator>(inputs, casted, stats);
	}

	return node;
}

autoDiff::NodePtr getOperationByDesc(const std::string& description,
									 const std::vector<autoDiff::NodePtr>& inputs)
{
	using namespace autoDiff;

	if(description == "LN")
	{
		return ops::naturalLog(inputs.front());
	}

	if(description == "RELU")
	{
		return ops::relu(inputs.front());
	}

	if(description == "SIGMOID")
	{
		return ops::sigmoid(inputs.front());
	}

	if(description == "MULTIPLY")
	{
		return ops::multiply(inputs.front(), inputs.back());
	}

	if(description == "ADD")
	{
		return ops::add(inputs.front(), inputs.back());
	}

	if(description == "SUBTRACT")
	{
		return ops::subtract(inputs.front(), inputs.back());
	}

	if(description == "MATMUL")
	{
		return ops::matmul(inputs.front(), inputs.back());
	}

	if(description == "DIVIDE")
	{
		return ops::divide(inputs.front(), inputs.back());
	}

	LOG_ERROR("TestComputationGraph", "Unknown operation type: " << description);
	return {};
}

/**
 * @brief Generates a tree of nodes based on given config.
 *
 * @param config Vector of pairs: <name of new node, operation and names of its inputs>.
 * @return Pair: <constructed nodes map, pairs of [parent, child] relations between the nodes>.
 */
std::pair<std::map<std::string, autoDiff::NodePtr>, std::shared_ptr<OperatorStats>>
constructTree(const std::vector<std::pair<std::string, std::string>>& config)
{
	using namespace autoDiff;

	std::map<std::string, NodePtr> nodes;
	const auto stats = std::make_shared<OperatorStats>();

	auto createNode = [&nodes, &stats](const std::string& name, const std::string& recipeStr) -> NodePtr
	{
		auto recipe = recipeStr;

		const auto oper = recipe.substr(0, recipe.find('_'));
		recipe = recipe.substr(oper.size() + 1);

		if(oper == "VARIABLE")
		{
			recipe = recipe.substr(1, recipe.size() - 2);

			std::istringstream iss(recipe);
			std::string shapeItem;
			std::vector<size_t> shape;

			while(std::getline(iss, shapeItem, ','))
			{
				shape.push_back(std::stoull(shapeItem));
			}

			const mlCore::Tensor value(shape);
			auto node = std::make_shared<Variable>(value);

			node->setName(name);

			return node;
		}

		if(oper == "PLACEHOLDER")
		{
			recipe = recipe.substr(1, recipe.size() - 2);

			std::istringstream iss(recipe);
			std::string shapeItem;
			std::vector<size_t> shape;

			while(std::getline(iss, shapeItem, ','))
			{
				shape.push_back(std::stoull(shapeItem));
			}

			auto node = std::make_shared<Placeholder>(std::make_shared<mlCore::Tensor>(shape));

			node->setName(name);

			return node;
		}

		std::istringstream iss(recipe);
		std::vector<std::string> inputNames;
		std::string inputName;

		while(std::getline(iss, inputName, '_'))
		{
			inputNames.push_back(inputName);
		}

		std::vector<NodePtr> inputs;
		std::transform(inputNames.cbegin(),
					   inputNames.cend(),
					   std::back_inserter(inputs),
					   [&nodes](const auto& inputName) { return nodes.at(inputName); });

		NodePtr binaryOper = getOperationByDesc(oper, inputs);

		binaryOper->setName(name);

		return wrapNode(inputs, binaryOper, stats);
	};

	for(const auto& [name, recipe] : config)
	{
		nodes[name] = createNode(name, recipe);
	}

	return {nodes, stats};
}
} // namespace mlCoreTests

#endif