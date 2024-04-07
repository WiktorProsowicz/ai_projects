/**********************
 * Test suite for 'ai_projects'
 *
 * Copyright (c) 2023
 *
 * by Wiktor Prosowicz
 **********************/
#include <sstream>

#include <AutoDiff/ComputationGraph.h>
#include <AutoDiff/Operations.h>
#include <MLCore/TensorInitializers/GaussianInitializer.hpp>
#include <MLCore/TensorInitializers/RangeTensorInitializer.hpp>
#include <gtest/gtest.h>

namespace
{
/*****************************
 *
 * Common data structures
 *
 *****************************/

class OperatorDecorator;
enum class WrapperLogChannel : uint8_t;

autoDiff::NodePtr
wrapNode(const autoDiff::NodePtr& node,
		 const std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>>& logs);

std::vector<autoDiff::NodePtr>
wrapNodes(const std::vector<autoDiff::NodePtr>& node,
		  const std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>>& logs);

// Enum specifying type of log made by computation graph node wrappers.
enum class WrapperLogChannel : uint8_t
{
	UPDATE_VALUE,
	COMPUTE_DERIVATIVE
};

class OperatorDecorator : public autoDiff::Operator
{

public:
	OperatorDecorator(const autoDiff::OperatorPtr& oper,
					  const std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>>& logs)
		: Operator(wrapNodes(oper->getInputs(), logs))
		, _oper(oper)
		, _logs(logs)

	{
		setName(oper->getName());
	}

	const mlCore::Tensor& getValue() const override
	{
		return _oper->getValue();
	}

	autoDiff::NodePtr copy() const override
	{
		return _oper->copy();
	}

	void updateValue() override
	{
		_oper->updateValue();

		for(const auto& input : getInputs())
		{
			(*_logs)[WrapperLogChannel::UPDATE_VALUE].push_back(input->getName() + " -> " + getName());
		}
	}

	std::vector<mlCore::Tensor> computeDerivative(const mlCore::Tensor& outerDerivative) const override
	{
		for(const auto& input : getInputs())
		{
			(*_logs)[WrapperLogChannel::COMPUTE_DERIVATIVE].push_back(input->getName() + " <- " + getName());
		}

		return _oper->computeDerivative(outerDerivative);
	}

	std::vector<mlCore::Tensor> computeDirectDerivative() const override
	{
		return _oper->computeDirectDerivative();
	}

	const std::vector<size_t>& getOutputShape() const override
	{
		return _oper->getOutputShape();
	}

private:
	autoDiff::OperatorPtr _oper;
	std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>> _logs;
};

struct BackPropagationConfig
{
	autoDiff::NodePtr tree;
	std::map<WrapperLogChannel, std::vector<std::string>> expectedLogs;
	std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>> sharedLogs;
};

/*****************************
 *
 * Common functions
 *
 *****************************/

/// Decorates input `node` or returns it if it is not of operator type.
autoDiff::NodePtr wrapNode(const autoDiff::NodePtr& node,
						   const std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>>& logs)
{
	using namespace autoDiff;

	if(const auto casted = std::dynamic_pointer_cast<Operator>(node))
	{
		return std::make_shared<OperatorDecorator>(casted, logs);
	}

	return node;
}

std::vector<autoDiff::NodePtr>
wrapNodes(const std::vector<autoDiff::NodePtr>& node,
		  const std::shared_ptr<std::map<WrapperLogChannel, std::vector<std::string>>>& logs)
{
	using namespace autoDiff;

	std::vector<NodePtr> wrappedNodes;

	std::for_each(node.cbegin(),
				  node.cend(),
				  [&wrappedNodes, &logs](const auto& node) { wrappedNodes.push_back(wrapNode(node, logs)); });

	return wrappedNodes;
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
 * @return Pair: <head of the constructed tree, pairs of [parent, child] relations between the nodes>.
 */
std::pair<autoDiff::NodePtr, std::set<std::pair<autoDiff::NodePtr, autoDiff::NodePtr>>>
constructTree(const std::vector<std::pair<std::string, std::string>>& config)
{
	using namespace autoDiff;

	std::set<std::pair<NodePtr, NodePtr>> relations;
	std::map<std::string, NodePtr> nodes;

	auto createNode = [&nodes, &relations](const std::string& name, const std::string& recipeStr) -> NodePtr
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

		std::vector<NodePtr> inputs;
		std::transform(inputNames.cbegin(),
					   inputNames.cend(),
					   std::back_inserter(inputs),
					   [&nodes](const auto& inputName) { return nodes.at(inputName); });

		NodePtr binaryOper = getOperationByDesc(oper, inputs);

		binaryOper->setName(name);

		std::for_each(inputs.cbegin(),
					  inputs.cend(),
					  [&relations, &binaryOper](const auto& input) { relations.emplace(binaryOper, input); });

		return binaryOper;

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
		_graph = std::make_shared<autoDiff::ComputationGraph>();
	}

	/**
	 * @brief Traverses tree and returns all unique nodes.
	 *
	 * @param root Root node of the traversed tree.
	 * @return All of the nodes present in the tree.
	 */
	static std::set<autoDiff::NodePtr> _flattenTree(const autoDiff::NodePtr& root)
	{
		using namespace autoDiff;

		std::set<NodePtr> nodes;

		std::function<void(const NodePtr)> flattenTree;

		flattenTree = [&nodes, &flattenTree](const NodePtr& node)
		{
			nodes.insert(node);

			if(const auto casted = std::dynamic_pointer_cast<Operator>(node))
			{
				for(const auto& input : casted->getInputs())
				{
					flattenTree(input);
				}
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
	static std::map<autoDiff::PlaceholderPtr, mlCore::Tensor>
	_createFeedMap(const std::set<autoDiff::PlaceholderPtr>& inputs)
	{
		using namespace autoDiff;

		std::map<PlaceholderPtr, mlCore::Tensor> feedMap;

		for(const auto& input : inputs)
		{
			const mlCore::tensorInitializers::GaussianInitializer<double> initializer;
			mlCore::Tensor inputTensor(input->getValue().shape());

			inputTensor.fill(initializer);

			feedMap.emplace(input, std::move(inputTensor));
		}

		return feedMap;
	}

	/**
	 * @brief Traverses the graph and checks whether the nodes are connected as expected.
	 *
	 * @param expectedRelations Set of (parent, child) pairs containing ids of graph nodes from root
	 * perspective.
	 * @param root Node from whose perspective the relations are collected.
	 */
	static void _checkNodesRelationships(
		const std::set<std::pair<autoDiff::NodePtr, autoDiff::NodePtr>>& expectedRelations,
		const autoDiff::NodePtr& root)
	{
		using namespace autoDiff;

		std::set<std::pair<NodePtr, NodePtr>> collectedRelations;

		std::function<void(const NodePtr&)> traverseTree;

		traverseTree = [&traverseTree, &collectedRelations](const NodePtr& node)
		{
			if(const auto casted = std::dynamic_pointer_cast<Operator>(node))
			{
				for(const auto& input : casted->getInputs())
				{
					collectedRelations.emplace(casted, input);

					traverseTree(input);
				}
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
	 * @brief Constructs an instance of ComputationGraph and runs it in both directions, collecting gradients
	 * and checking whether all of the value-updating and derivative-computing operations are run.
	 *
	 * @param tree Root node of the tree to construct the graph from.
	 */
	void _checkBackPropagation(const BackPropagationConfig& config)
	{
		using namespace autoDiff;

		const auto nodes = _flattenTree(config.tree);

		_graph = std::make_shared<ComputationGraph>();

		_graph->activate();

		for(const auto& node : nodes)
		{
			_graph->addNode(node);
		}

		_graph->forwardPass();
		_graph->computeGradients(config.tree);

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
	void _performGradientDescent(const autoDiff::NodePtr& tree,
								 const std::set<autoDiff::NodePtr>& trainableWeights,
								 const autoDiff::PlaceholderPtr& input)
	{
		using namespace autoDiff;

		_graph = std::make_shared<ComputationGraph>();
		_graph->activate();

		const auto nodes = _flattenTree(tree);

		for(const auto& node : nodes)
		{
			_graph->addNode(node);
		}

		for(uint8_t batchNumber = 0; batchNumber < 16; batchNumber++)
		{
			_graph->clearGradients();

			for(uint8_t loopCount = 0; loopCount < 32; loopCount++)
			{
				const auto feedMap = _createFeedMap({input});

				std::cout << feedMap.at(input) << '\n';

				_graph->forwardPass(feedMap);
			}

			_graph->computeGradients(tree);

			for(const auto& weight : trainableWeights)
			{
				const auto& derivative = _graph->getGradient(weight);

				if(const auto castedVar = std::dynamic_pointer_cast<Variable>(weight))
				{
					castedVar->setValue(castedVar->getValue() - (derivative * .1));
				}
			}
		}
	}

protected:
	std::shared_ptr<autoDiff::ComputationGraph> _graph = nullptr;
};

/*****************************
 *
 * Particular test calls
 *
 *****************************/

TEST_F(TestComputationGraph, testGraphStructureBuilding)
{
	using namespace autoDiff;

	const std::vector<std::pair<std::string, std::string>> config{{"0", "VARIABLE_(5,5)"},
																  {"1", "VARIABLE_(5,5)"},
																  {"2", "MULTIPLY_0_0"},
																  {"3", "ADD_2_1"},
																  {"4", "LN_3"},
																  {"5", "VARIABLE_(5,5)"},
																  {"6", "RELU_5"},
																  {"7", "SIGMOID_6"},
																  {"8", "VARIABLE_(5,5)"},
																  {"9", "MULTIPLY_7_8"},
																  {"10", "SUBTRACT_4_9"},
																  {"11", "VARIABLE_(5,5)"},
																  {"12", "VARIABLE_(5,5)"},
																  {"13", "MATMUL_11_12"},
																  {"14", "DIVIDE_10_13"}};

	auto [tree, relations] = constructTree(config);

	_checkNodesRelationships(relations, tree);
}

TEST_F(TestComputationGraph, testBackPropagation)
{
	using namespace autoDiff;

	const std::vector<std::pair<std::string, std::string>> treeConstructionConfig{
		{"Input", "PLACEHOLDER_(256,1)"},
		{"L1W", "VARIABLE_(200,256)"},
		{"L1B", "VARIABLE_(200,1)"},
		{"Layer1", "MATMUL_L1W_Input"},
		{"Layer1biased", "ADD_Layer1_L1B"},
		{"Layer1Act", "RELU_Layer1biased"},
		{"L2W", "VARIABLE_(200,200)"},
		{"L2B", "VARIABLE_(200,1)"},
		{"Layer2", "MATMUL_L2W_Layer1Act"},
		{"Layer2biased", "ADD_Layer2_L2B"},
		{"Layer2Act", "SIGMOID_Layer2biased"},
		{"L3W", "VARIABLE_(1,200)"},
		{"L3B", "VARIABLE_(1,1)"},
		{"Layer3", "MATMUL_L3W_Layer2Act"},
		{"Layer3biased", "ADD_Layer3_L3B"},
		{"Layer3Act", "SIGMOID_Layer3biased"},
		{"OutputLayer", "LN_Layer3Act"}};

	auto [tree, relations] = constructTree(treeConstructionConfig);

	auto wrappersLogs = std::make_shared<std::map<WrapperLogChannel, std::vector<std::string>>>();

	const std::map<WrapperLogChannel, std::vector<std::string>> expectedLogs{
		{WrapperLogChannel::UPDATE_VALUE,
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

	const BackPropagationConfig config{
		.tree = wrapNode(tree, wrappersLogs), .expectedLogs = expectedLogs, .sharedLogs = wrappersLogs};

	_checkBackPropagation(config);
}

TEST_F(TestComputationGraph, testGradientDescentSimulation)
{
	using namespace autoDiff;

	const std::vector<std::pair<std::string, std::string>> treeConfig{{"Input", "PLACEHOLDER_(256,1)"},
																	  {"L1W", "VARIABLE_(200,256)"},
																	  {"L1B", "VARIABLE_(200,1)"},
																	  {"Layer1", "MATMUL_L1W_Input"},
																	  {"Layer1biased", "ADD_Layer1_L1B"},
																	  {"Layer1Act", "RELU_Layer1biased"},
																	  {"L2W", "VARIABLE_(200,200)"},
																	  {"L2B", "VARIABLE_(200,1)"},
																	  {"Layer2", "MATMUL_L2W_Layer1Act"},
																	  {"Layer2biased", "ADD_Layer2_L2B"},
																	  {"Layer2Act", "SIGMOID_Layer2biased"},
																	  {"L3W", "VARIABLE_(1,200)"},
																	  {"L3B", "VARIABLE_(1,1)"},
																	  {"Layer3", "MATMUL_L3W_Layer2Act"},
																	  {"Layer3biased", "ADD_Layer3_L3B"},
																	  {"Layer3Act", "SIGMOID_Layer3biased"}};

	auto [tree, relations] = constructTree(treeConfig);

	auto wrappersLogs = std::make_shared<std::map<WrapperLogChannel, std::vector<std::string>>>();

	auto wrappedTree = wrapNode(tree, wrappersLogs);

	std::set<NodePtr> trainableWeights;
	PlaceholderPtr input;

	for(const auto& node : _flattenTree(wrappedTree))
	{
		constexpr static std::array<const char*, 6> kTrainableNames = {
			"L1W", "L1B", "L2W", "L2B", "L3W", "L3B"};

		if(std::find(kTrainableNames.cbegin(), kTrainableNames.cend(), node->getName()) !=
		   kTrainableNames.cend())
		{
			trainableWeights.insert(node);
		}

		if(node->getName() == "Input")
		{
			input = std::dynamic_pointer_cast<Placeholder>(node);
		}
	}

	ASSERT_NO_THROW(_performGradientDescent(wrappedTree, trainableWeights, input));
}

} // namespace
