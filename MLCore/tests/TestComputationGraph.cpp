/**********************
 * Test suite for 'ai_projects'
 * 
 * Copyright (c) 2023
 * 
 * by Wiktor Prosowicz
 **********************/
#include <sstream>

#include <AutoDiff/ComputationGraph.h>

#include <AutoDiff/BinaryOperators/BinaryOperator.h>
#include <AutoDiff/UnaryOperators/UnaryOperator.h>

#include <AutoDiff/GraphOperations.h>

#include <gtest/gtest.h>

namespace
{
/*****************************
 * 
 * Common data structures
 * 
 *****************************/

class BinaryOperatorDecorator;
class UnaryOperatorDecorator;

mlCore::autoDiff::NodePtr wrapNode(mlCore::autoDiff::NodePtr node);

class BinaryOperatorDecorator : public mlCore::autoDiff::binaryOperators::BinaryOperator
{

public:
	BinaryOperatorDecorator(const mlCore::autoDiff::binaryOperators::BinaryOperatorPtr oper)
		: oper_(oper)
		, BinaryOperator(wrapNode(oper->getInputs().first), wrapNode(oper->getInputs().second))
	{
		setName(oper->getName() + "Wrapper");
	}

	void updateValue() override
	{
		oper_->updateValue();
		value_ = oper_->getValue();

		std::cout << "Updated value of '" << getName() << "'" << std::endl;
	}

	std::pair<mlCore::Tensor, mlCore::Tensor> computeDerivative(const mlCore::Tensor& outerDerivative) const override
	{
		return oper_->computeDerivative(outerDerivative);
	}

	virtual std::pair<mlCore::Tensor, mlCore::Tensor> computeDirectDerivative() const override
	{
		return oper_->computeDirectDerivative();
	}

private:
	mlCore::autoDiff::binaryOperators::BinaryOperatorPtr oper_;
};

class UnaryOperatorDecorator : public mlCore::autoDiff::unaryOperators::UnaryOperator
{

public:
	UnaryOperatorDecorator(const mlCore::autoDiff::unaryOperators::UnaryOperatorPtr oper)
		: oper_(oper)
		, UnaryOperator(wrapNode(oper->getInput()))
	{
		setName(oper->getName() + "Wrapper");
	}

	void updateValue() override
	{
		oper_->updateValue();
		value_ = oper_->getValue();

		std::cout << "Updated value of '" << getName() << "'" << std::endl;
	}

	mlCore::Tensor computeDerivative(const mlCore::Tensor& outerDerivative) const override
	{
		return oper_->computeDerivative(outerDerivative);
	}

	mlCore::Tensor computeDirectDerivative() const override
	{
		return oper_->computeDirectDerivative();
	}

private:
	mlCore::autoDiff::unaryOperators::UnaryOperatorPtr oper_;
};

/// Decorates input `node` or returns it if it is not of operator type.
mlCore::autoDiff::NodePtr wrapNode(mlCore::autoDiff::NodePtr node)
{
	using namespace mlCore::autoDiff;

	if(const auto casted = std::dynamic_pointer_cast<binaryOperators::BinaryOperator>(node))
	{
		return std::make_shared<BinaryOperatorDecorator>(casted);
	}
	else if(const auto casted = std::dynamic_pointer_cast<unaryOperators::UnaryOperator>(node))
	{
		return std::make_shared<UnaryOperatorDecorator>(casted);
	}
	else
	{
		return node;
	}
}

/*****************************
 * 
 * Common functions
 * 
 *****************************/

/**
 * @brief Generates a tree of nodes based on given config.
 * 
 * @param config Vector of pairs: <name of new node, operation and names of its inputs>.
 * @return Pair: <head of the constructed tree, pairs of [parent, child] relations between the nodes>.
 */
std::pair<mlCore::autoDiff::NodePtr, std::set<std::pair<uint64_t, uint64_t>>>
constructTree(const std::vector<std::pair<std::string, std::string>>& config)
{
	using namespace mlCore::autoDiff;

	std::set<std::pair<uint64_t, uint64_t>> relations;
	std::map<std::string, NodePtr> nodes;

	auto createNode = [&nodes, &relations](const std::string& name, const std::string& recipeStr) -> NodePtr {
		auto recipe = recipeStr;

		const auto oper = recipe.substr(0, recipe.find('_'));
		recipe = recipe.substr(oper.size() + 1);

		if(oper == "NODE")
		{
			recipe = recipe.substr(1, recipe.size() - 2);

			std::istringstream iss(recipe);
			std::string shapeItem;
			std::vector<size_t> shape;

			while(std::getline(iss, shapeItem, ','))
			{
				shape.push_back(std::stoll(shapeItem));
			}

			mlCore::Tensor value(shape);
			auto node = std::make_shared<Node>(value);

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

		if(inputNames.size() == 2)
		{
			auto lhs = nodes.at(inputNames[0]);
			auto rhs = nodes.at(inputNames[1]);

			NodePtr binaryOper;

			if(oper == "MULTIPLY")
			{
				binaryOper = binaryOperations::multiply(lhs, rhs);
			}
			else if(oper == "ADD")
			{
				binaryOper = binaryOperations::add(lhs, rhs);
			}
			else if(oper == "POWER")
			{
				binaryOper = binaryOperations::power(lhs, rhs);
			}
			else if(oper == "SUBTRACT")
			{
				binaryOper = binaryOperations::subtract(lhs, rhs);
			}
			else if(oper == "MATMUL")
			{
				binaryOper = binaryOperations::matmul(lhs, rhs);
			}
			else if(oper == "DIVIDE")
			{
				binaryOper = binaryOperations::divide(lhs, rhs);
			}
			else
			{
				LOG_ERROR("TestComputationGraph", "Unknown operation type: " << oper);
			}

			binaryOper->setName(name);

			relations.emplace(binaryOper->getIndex(), lhs->getIndex());
			relations.emplace(binaryOper->getIndex(), rhs->getIndex());

			return binaryOper;
		}
		else if(inputNames.size() == 1)
		{
			auto input = nodes.at(inputNames[0]);

			NodePtr unaryOper;

			if(oper == "LN")
			{
				unaryOper = unaryOperations::ln(input);
			}
			else if(oper == "RELU")
			{
				unaryOper = nodesActivations::relu(input);
			}
			else if(oper == "SIGMOID")
			{
				unaryOper = nodesActivations::sigmoid(input);
			}
			else
			{
				LOG_ERROR("TestComputationGraph", "Unknown operation type: " << oper);
			}

			unaryOper->setName(name);

			relations.emplace(unaryOper->getIndex(), input->getIndex());

			return unaryOper;
		}
		else
		{
			LOG_ERROR("TestComputationGraph", "Unhandled number of inputs: " << inputNames.size());
			return {};
		}
	};

	for(const auto& [name, recipe] : config)
	{
		nodes[name] = createNode(name, recipe);
	}

	return {nodes[config.crbegin()->first], relations};
}

/*****************************
 * 
 * Test Fixture
 * 
 *****************************/
class TestComputationGraph : public testing::Test
{
protected:
	void SetUp() override
	{
		graph_ = std::make_shared<mlCore::autoDiff::ComputationGraph>();
	}

	/**
     * @brief Traverses the graph and checks whether the nodes are connected as expected.
     * 
     * @param expectedRelations Set of (parent, child) pairs containing ids of graph nodes from root perspective.
     * @param root Node from whose perspective the relations are collected.
     */
	static void checkNodesRelationships(const std::set<std::pair<uint64_t, uint64_t>>& expectedRelations,
										mlCore::autoDiff::NodePtr root)
	{
		using namespace mlCore::autoDiff;

		std::set<std::pair<uint64_t, uint64_t>> collectedRelations;

		std::function<void(const NodePtr)> traverseTree;

		traverseTree = [&traverseTree, &collectedRelations](const NodePtr node) {
			if(const auto casted = std::dynamic_pointer_cast<unaryOperators::UnaryOperator>(node))
			{
				collectedRelations.emplace(casted->getIndex(), casted->getInput()->getIndex());

				traverseTree(casted->getInput());
			}
			else if(const auto casted = std::dynamic_pointer_cast<binaryOperators::BinaryOperator>(node))
			{
				const auto [leftInput, rightInput] = casted->getInputs();

				collectedRelations.emplace(casted->getIndex(), leftInput->getIndex());

				collectedRelations.emplace(casted->getIndex(), rightInput->getIndex());

				traverseTree(leftInput);
				traverseTree(rightInput);
			}
		};

		traverseTree(root);

		ASSERT_EQ(collectedRelations.size(), expectedRelations.size());

		for(const auto& parentChildPair : expectedRelations)
		{
			ASSERT_TRUE(std::find(collectedRelations.cbegin(), collectedRelations.cend(), parentChildPair) !=
						collectedRelations.end());
		}
	}

	void checkBackPropagation(const mlCore::autoDiff::NodePtr tree)
	{
		using namespace mlCore::autoDiff;

		graph_ = std::make_shared<ComputationGraph>();

		std::set<NodePtr> nodes;

		std::function<void(const NodePtr)> flattenTree;

		flattenTree = [&nodes, &flattenTree](const NodePtr node) {
			nodes.insert(node);

			if(const auto casted = std::dynamic_pointer_cast<unaryOperators::UnaryOperator>(node))
			{
				flattenTree(casted->getInput());
			}
			else if(const auto casted = std::dynamic_pointer_cast<binaryOperators::BinaryOperator>(node))
			{
				const auto [leftInput, rightInput] = casted->getInputs();

				flattenTree(leftInput);
				flattenTree(rightInput);
			}
		};

		flattenTree(tree);

		graph_->activate();

		for(const auto& node : nodes)
		{
			graph_->addNode(node);
		}

		graph_->forwardPass({});
		graph_->computeGradients(tree);
	}

protected:
	std::shared_ptr<mlCore::autoDiff::ComputationGraph> graph_ = nullptr;
};

/*****************************
 * 
 * Particular test calls
 * 
 *****************************/

TEST_F(TestComputationGraph, testGraphStructureBuilding)
{
	using namespace mlCore::autoDiff;

	const std::vector<std::pair<std::string, std::string>> config{{"0", "NODE_(5,5)"},
																  {"1", "NODE_(5,5)"},
																  {"2", "MULTIPLY_0_0"},
																  {"3", "ADD_2_1"},
																  {"4", "LN_3"},
																  {"5", "NODE_(5,5)"},
																  {"6", "RELU_5"},
																  {"7", "SIGMOID_6"},
																  {"8", "NODE_(5,5)"},
																  {"9", "POWER_7_8"},
																  {"10", "SUBTRACT_4_9"},
																  {"11", "NODE_(5,5)"},
																  {"12", "NODE_(5,5)"},
																  {"13", "MATMUL_11_12"},
																  {"14", "DIVIDE_10_13"}};

	auto [tree, relations] = constructTree(config);

	checkNodesRelationships(relations, tree);
}

TEST_F(TestComputationGraph, testBackPropagation)
{
	using namespace mlCore::autoDiff;

	const std::vector<std::pair<std::string, std::string>> config{{"Input", "NODE_(20,1)"},
																  {"L1W", "NODE_(10,20)"},
																  {"L1B", "NODE_(10,1)"},
																  {"Layer1", "MATMUL_L1W_Input"},
																  {"Layer1biased", "ADD_Layer1_L1B"},
																  {"Layer1Act", "RELU_Layer1biased"},
																  {"L2W", "NODE_(10,10)"},
																  {"L2B", "NODE_(10,1)"},
																  {"Layer2", "MATMUL_L2W_Layer1Act"},
																  {"Layer2biased", "ADD_Layer2_L2B"},
																  {"Layer2Act", "SIGMOID_Layer2biased"},
																  {"L3W", "NODE_(1,10)"},
																  {"L3B", "NODE_(1,1)"},
																  {"Layer3", "MATMUL_L3W_Layer2Act"},
																  {"Layer3biased", "ADD_Layer3_L3B"},
																  {"Layer3Act", "LN_Layer3biased"}};

	auto [tree, relations] = constructTree(config);

	checkBackPropagation(wrapNode(tree));
}

} // namespace