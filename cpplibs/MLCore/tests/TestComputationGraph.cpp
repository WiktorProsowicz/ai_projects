/**********************
 * Test suite for 'ai_projects'
 *
 * Copyright (c) 2023
 *
 * by Wiktor Prosowicz
 **********************/
#include <AutoDiff/ComputationGraph.h>

#include <sstream>

#include <gtest/gtest.h>

#include <AutoDiff/BinaryOperators/BinaryOperator.h>
#include <AutoDiff/UnaryOperators/UnaryOperator.h>
#include <AutoDiff/GraphOperations.h>
#include <MLCore/TensorInitializers/RangeTensorInitializer.hpp>
#include <MLCore/TensorInitializers/GaussianInitializer.hpp>

namespace
{
/*****************************
 *
 * Common data structures
 *
 *****************************/

class BinaryOperatorDecorator;
class UnaryOperatorDecorator;
enum class WrapperLogChannel : uint8_t;

mlCore::autoDiff::NodePtr wrapNode(mlCore::autoDiff::NodePtr node,
								   std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>> logs);

// Enum specifying type of log made by computation graph node wrappers.
enum class WrapperLogChannel : uint8_t
{
	UPDATE_VALUE,
	COMPUTE_DERIVATIVE
};

class BinaryOperatorDecorator : public mlCore::autoDiff::binaryOperators::BinaryOperator
{

public:
	BinaryOperatorDecorator(const mlCore::autoDiff::binaryOperators::BinaryOperatorPtr oper,
							const std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>> logs)
		: BinaryOperator(wrapNode(oper->getInputs().first, logs), wrapNode(oper->getInputs().second, logs))
		, oper_(oper)
		, logs_(logs)

	{
		setName(oper->getName());
	}

	void updateValue() override
	{
		oper_->updateValue();
		value_ = oper_->getValue();

		const auto& [lhs, rhs] = getInputs();

		(*logs_)[WrapperLogChannel::UPDATE_VALUE].push_back(lhs->getName() + " -> " + getName());
		(*logs_)[WrapperLogChannel::UPDATE_VALUE].push_back(rhs->getName() + " -> " + getName());
	}

	std::pair<mlCore::Tensor, mlCore::Tensor> computeDerivative(const mlCore::Tensor& outerDerivative) const override
	{

		const auto& [lhs, rhs] = getInputs();

		(*logs_)[WrapperLogChannel::COMPUTE_DERIVATIVE].push_back(lhs->getName() + " <- " + getName());
		(*logs_)[WrapperLogChannel::COMPUTE_DERIVATIVE].push_back(rhs->getName() + " <- " + getName());

		return oper_->computeDerivative(outerDerivative);
	}

	std::pair<mlCore::Tensor, mlCore::Tensor> computeDirectDerivative() const override
	{
		return oper_->computeDirectDerivative();
	}

private:
	mlCore::autoDiff::binaryOperators::BinaryOperatorPtr oper_;
	std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>> logs_;
};

class UnaryOperatorDecorator : public mlCore::autoDiff::unaryOperators::UnaryOperator
{
public:
	UnaryOperatorDecorator(const mlCore::autoDiff::unaryOperators::UnaryOperatorPtr oper,
						   const std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>> logs)
		: UnaryOperator(wrapNode(oper->getInput(), logs))
		, oper_(oper)
		, logs_(logs)

	{
		setName(oper->getName());
	}

	void updateValue() override
	{
		oper_->updateValue();
		value_ = oper_->getValue();

		(*logs_)[WrapperLogChannel::UPDATE_VALUE].push_back(getInput()->getName() + " -> " + getName());
	}

	mlCore::Tensor computeDerivative(const mlCore::Tensor& outerDerivative) const override
	{
		(*logs_)[WrapperLogChannel::COMPUTE_DERIVATIVE].push_back(getInput()->getName() + " <- " + getName());

		return oper_->computeDerivative(outerDerivative);
	}

	mlCore::Tensor computeDirectDerivative() const override
	{
		return oper_->computeDirectDerivative();
	}

private:
	mlCore::autoDiff::unaryOperators::UnaryOperatorPtr oper_;
	std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>> logs_;
};

struct BackPropagationConfig
{
	mlCore::autoDiff::NodePtr tree;
	std::map<WrapperLogChannel, std::vector<std::string>> expectedLogs;
	std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>> sharedLogs;
};

/*****************************
 *
 * Common functions
 *
 *****************************/

/// Decorates input `node` or returns it if it is not of operator type.
mlCore::autoDiff::NodePtr wrapNode(mlCore::autoDiff::NodePtr node,
								   std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>> logs)
{
	using namespace mlCore::autoDiff;

	if(const auto casted = std::dynamic_pointer_cast<binaryOperators::BinaryOperator>(node))
	{
		return std::make_shared<BinaryOperatorDecorator>(casted, logs);
	}

	if(const auto casted = std::dynamic_pointer_cast<unaryOperators::UnaryOperator>(node))
	{
		return std::make_shared<UnaryOperatorDecorator>(casted, logs);
	}

	return node;
}

mlCore::autoDiff::NodePtr getUnaryOperationByDesc(const std::string& description, const mlCore::autoDiff::NodePtr& input)
{
	using namespace mlCore::autoDiff;

	if(description == "LN")
	{
		return unaryOperations::ln(input);
	}

	if(description == "RELU")
	{
		return nodesActivations::relu(input);
	}

	if(description == "SIGMOID")
	{
		return nodesActivations::sigmoid(input);
	}

	LOG_ERROR("TestComputationGraph", "Unknown operation type: " << description);
	return {};
}

mlCore::autoDiff::NodePtr getBinaryOperationByDesc(const std::string& description,
												   const mlCore::autoDiff::NodePtr& lhs,
												   const mlCore::autoDiff::NodePtr& rhs)
{
	using namespace mlCore::autoDiff;

	if(description == "MULTIPLY")
	{
		return binaryOperations::multiply(lhs, rhs);
	}
	if(description == "ADD")
	{
		return binaryOperations::add(lhs, rhs);
	}
	if(description == "POWER")
	{
		return binaryOperations::power(lhs, rhs);
	}
	if(description == "SUBTRACT")
	{
		return binaryOperations::subtract(lhs, rhs);
	}
	if(description == "MATMUL")
	{
		return binaryOperations::matmul(lhs, rhs);
	}
	if(description == "DIVIDE")
	{
		return binaryOperations::divide(lhs, rhs);
	}

	LOG_ERROR("TestComputationGraph", "Unknown operation type: " << description);
	return {};
}

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
				shape.push_back(std::stoull(shapeItem));
			}

			mlCore::Tensor value(shape);
			auto node = std::make_shared<Node>(value);

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

			auto node = std::make_shared<Placeholder>(shape);

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

			NodePtr binaryOper = getBinaryOperationByDesc(oper, lhs, rhs);

			binaryOper->setName(name);

			relations.emplace(binaryOper->getIndex(), lhs->getIndex());
			relations.emplace(binaryOper->getIndex(), rhs->getIndex());

			return binaryOper;
		}

		if(inputNames.size() == 1)
		{
			auto input = nodes.at(inputNames[0]);

			NodePtr unaryOper = getUnaryOperationByDesc(oper, input);

			unaryOper->setName(name);

			relations.emplace(unaryOper->getIndex(), input->getIndex());

			return unaryOper;
		}

		LOG_ERROR("TestComputationGraph", "Unhandled number of inputs: " << inputNames.size());
		return {};
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
     * @brief Traverses tree and returns all unique nodes.
     *
     * @param root Root node of the traversed tree.
     * @return All of the nodes present in the tree.
     */
	static std::set<mlCore::autoDiff::NodePtr> flattenTree(const mlCore::autoDiff::NodePtr root)
	{
		using namespace mlCore::autoDiff;

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

		flattenTree(root);

		return nodes;
	}

	/**
	 * @brief Creates a feed map that can be used in forward-pass of ComputationGraph.
	 *
	 * @param inputs Placeholders to assign input tensors to.
	 * @return Created feed map with random generated inputs.
	 */
	static std::map<mlCore::autoDiff::PlaceholderPtr, mlCore::Tensor>
	createFeedMap(const std::set<mlCore::autoDiff::PlaceholderPtr>& inputs)
	{
		using namespace mlCore::autoDiff;

		std::map<PlaceholderPtr, mlCore::Tensor> feedMap;

		for(const auto& input : inputs)
		{
			mlCore::tensorInitializers::GaussianInitializer<double> initializer;
			mlCore::Tensor inputTensor(input->getValue().shape());

			inputTensor.fill(initializer);

			feedMap.emplace(input, std::move(inputTensor));
		}

		return feedMap;
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

	/**
     * @brief Constructs an instance of ComputationGraph and runs it in both directions, collecting gradients and checking whether all of the value-updating
     * and derivative-computing operations are run.
     *
     * @param tree Root node of the tree to construct the graph from.
     */
	void checkBackPropagation(const BackPropagationConfig& config)
	{
		using namespace mlCore::autoDiff;

		const auto nodes = flattenTree(config.tree);

		graph_ = std::make_shared<ComputationGraph>();

		graph_->activate();

		for(const auto& node : nodes)
		{
			graph_->addNode(node);
		}

		graph_->forwardPass();
		graph_->computeGradients(config.tree);

		for(const auto& [logChannel, logs] : config.expectedLogs)
		{
			const auto& collectedLogs = config.sharedLogs->at(logChannel);

			ASSERT_EQ(logs.size(), collectedLogs.size());

			auto collectedLogsIter = collectedLogs.cbegin();

			for(const auto& expectedLog : logs)
			{
				ASSERT_STREQ(expectedLog.c_str(), collectedLogsIter->c_str());

				std::advance(collectedLogsIter, 1);
			}
		}
	}

	/**
	 * @brief Simulates computation graph forward pass and gradients updates.
	 *
	 * @param tree Root node of the nodes tree.
	 * @param trainableWeights Weights to be updated with gradient.
	 * @param input Input layer for feed map.
	 */
	void performGradientDescent(const mlCore::autoDiff::NodePtr tree,
								const std::set<mlCore::autoDiff::NodePtr>& trainableWeights,
								mlCore::autoDiff::PlaceholderPtr input)
	{
		using namespace mlCore::autoDiff;

		graph_ = std::make_shared<ComputationGraph>();
		graph_->activate();

		const auto nodes = flattenTree(tree);

		for(const auto& node : nodes)
		{
			graph_->addNode(node);
		}

		for(uint8_t batchNumber = 0; batchNumber < 16; batchNumber++)
		{
			graph_->clearGradients();

			for(uint8_t loopCount = 0; loopCount < 32; loopCount++)
			{
				const auto feedMap = createFeedMap({input});

				std::cout << feedMap.at(input) << std::endl;

				graph_->forwardPass(feedMap);
			}

			graph_->computeGradients(tree);

			for(auto weight : trainableWeights)
			{
				const auto& derivative = graph_->getGradientByNodeId(weight->getIndex());
				weight->getValue() -= (derivative * .1);
			}
		}
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

	const std::vector<std::pair<std::string, std::string>> treeConstructionConfig{{"Input", "PLACEHOLDER_(256,1)"},
																				  {"L1W", "NODE_(200,256)"},
																				  {"L1B", "NODE_(200,1)"},
																				  {"Layer1", "MATMUL_L1W_Input"},
																				  {"Layer1biased", "ADD_Layer1_L1B"},
																				  {"Layer1Act", "RELU_Layer1biased"},
																				  {"L2W", "NODE_(200,200)"},
																				  {"L2B", "NODE_(200,1)"},
																				  {"Layer2", "MATMUL_L2W_Layer1Act"},
																				  {"Layer2biased", "ADD_Layer2_L2B"},
																				  {"Layer2Act", "SIGMOID_Layer2biased"},
																				  {"L3W", "NODE_(1,200)"},
																				  {"L3B", "NODE_(1,1)"},
																				  {"Layer3", "MATMUL_L3W_Layer2Act"},
																				  {"Layer3biased", "ADD_Layer3_L3B"},
																				  {"Layer3Act", "SIGMOID_Layer3biased"},
																				  {"OutputLayer", "LN_Layer3Act"}};

	auto [tree, relations] = constructTree(treeConstructionConfig);

	auto wrappersLogs = std::make_shared<std::map<WrapperLogChannel, std::vector<std::string>>>();

	const std::map<WrapperLogChannel, std::vector<std::string>> expectedLogs{{WrapperLogChannel::UPDATE_VALUE,
																			  {"L1W -> Layer1",
																			   "Input -> Layer1",
																			   "Layer1 -> Layer1biased",
																			   "L1B -> Layer1biased",
																			   "Layer1biased -> Layer1Act",
																			   "L2W -> Layer2",
																			   "Layer1Act -> Layer2",
																			   "Layer2 -> Layer2biased",
																			   "L2B -> Layer2biased",
																			   "Layer2biased -> Layer2Act",
																			   "L3W -> Layer3",
																			   "Layer2Act -> Layer3",
																			   "Layer3 -> Layer3biased",
																			   "L3B -> Layer3biased",
																			   "Layer3biased -> Layer3Act",
																			   "Layer3Act -> OutputLayer"}},
																			 {WrapperLogChannel::COMPUTE_DERIVATIVE,
																			  {"Layer3Act <- OutputLayer",
																			   "Layer3biased <- Layer3Act",
																			   "Layer3 <- Layer3biased",
																			   "L3B <- Layer3biased",
																			   "L3W <- Layer3",
																			   "Layer2Act <- Layer3",
																			   "Layer2biased <- Layer2Act",
																			   "Layer2 <- Layer2biased",
																			   "L2B <- Layer2biased",
																			   "L2W <- Layer2",
																			   "Layer1Act <- Layer2",
																			   "Layer1biased <- Layer1Act",
																			   "Layer1 <- Layer1biased",
																			   "L1B <- Layer1biased",
																			   "L1W <- Layer1",
																			   "Input <- Layer1"}}};

	BackPropagationConfig config{.tree = wrapNode(tree, wrappersLogs), .expectedLogs = expectedLogs, .sharedLogs = wrappersLogs};

	checkBackPropagation(config);
}

TEST_F(TestComputationGraph, testGradientDescentSimulation)
{
	using namespace mlCore::autoDiff;

	const std::vector<std::pair<std::string, std::string>> treeConfig{{"Input", "PLACEHOLDER_(256,1)"},
																	  {"L1W", "NODE_(200,256)"},
																	  {"L1B", "NODE_(200,1)"},
																	  {"Layer1", "MATMUL_L1W_Input"},
																	  {"Layer1biased", "ADD_Layer1_L1B"},
																	  {"Layer1Act", "RELU_Layer1biased"},
																	  {"L2W", "NODE_(200,200)"},
																	  {"L2B", "NODE_(200,1)"},
																	  {"Layer2", "MATMUL_L2W_Layer1Act"},
																	  {"Layer2biased", "ADD_Layer2_L2B"},
																	  {"Layer2Act", "SIGMOID_Layer2biased"},
																	  {"L3W", "NODE_(1,200)"},
																	  {"L3B", "NODE_(1,1)"},
																	  {"Layer3", "MATMUL_L3W_Layer2Act"},
																	  {"Layer3biased", "ADD_Layer3_L3B"},
																	  {"Layer3Act", "SIGMOID_Layer3biased"}};

	auto [tree, relations] = constructTree(treeConfig);

	auto wrappersLogs = std::make_shared<std::map<WrapperLogChannel, std::vector<std::string>>>();

	auto wrappedTree = wrapNode(tree, wrappersLogs);

	std::set<NodePtr> trainableWeights;
	PlaceholderPtr input;

	for(const auto& node : flattenTree(wrappedTree))
	{
		constexpr static std::array<const char*, 6> kTrainableNames = {"L1W", "L1B", "L2W", "L2B", "L3W", "L3B"};

		if(std::find(kTrainableNames.cbegin(), kTrainableNames.cend(), node->getName()) != kTrainableNames.cend())
		{
			trainableWeights.insert(node);
		}

		if(node->getName() == "Input")
		{
			input = std::dynamic_pointer_cast<Placeholder>(node);
		}
	}

	performGradientDescent(wrappedTree, trainableWeights, input);
}

} // namespace
