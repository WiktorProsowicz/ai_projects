/**********************
 * Test suite for 'ai_projects'
 *
 * Copyright (c) 2024
 *
 * by Wiktor Prosowicz
 **********************/
#include <sstream>

#include <AutoDiff/ComputationGraph.h>
#include <AutoDiff/Operations.h>
#include <MLCore/TensorInitializers/GaussianInitializer.hpp>
#include <MLCore/TensorInitializers/RangeTensorInitializer.hpp>
#include <gtest/gtest.h>

#include "GraphConstructionUtils.hpp"

namespace
{
/*****************************
 *
 * Common data structures
 *
 *****************************/

struct BackPropagationConfig
{
	autoDiff::NodePtr tree;
	std::set<autoDiff::NodePtr> trainableWeights;
	std::shared_ptr<mlCoreTests::OperatorStats> operatorStats;
};

/*****************************
 *
 * Common functions
 *
 *****************************/

/**
 * @brief Creates a feed map that can be used in forward-pass of ComputationGraph.
 *
 * @param inputs Placeholders to assign input tensors to.
 * @return Created feed map with random generated inputs.
 */
std::map<autoDiff::PlaceholderPtr, std::shared_ptr<mlCore::Tensor>>
createFeedMap(const std::set<autoDiff::PlaceholderPtr>& inputs)
{
	using namespace autoDiff;

	std::map<PlaceholderPtr, std::shared_ptr<mlCore::Tensor>> feedMap;

	for(const auto& input : inputs)
	{
		const mlCore::tensorInitializers::GaussianInitializer<double> initializer;
		const auto inputTensor = std::make_shared<mlCore::Tensor>(input->getValue().shape());

		inputTensor->fill(initializer);

		feedMap.emplace(input, inputTensor);
	}

	return feedMap;
}

/**
 * @brief Traverses tree and returns all unique nodes.
 *
 * @param root Root node of the traversed tree.
 * @return All of the nodes present in the tree.
 */
std::set<autoDiff::NodePtr> flattenTree(const autoDiff::NodePtr& root)
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

std::set<autoDiff::NodePtr> getNodesByNames(const autoDiff::NodePtr& tree,
											const std::set<std::string>& nodeNames)
{
	const auto allNodes = flattenTree(tree);

	std::set<autoDiff::NodePtr> nodes;

	std::copy_if(allNodes.cbegin(),
				 allNodes.cend(),
				 std::inserter(nodes, nodes.end()),
				 [&nodeNames](const auto& node)
				 { return nodeNames.find(node->getName()) != nodeNames.cend(); });

	return nodes;
}

/*****************************
 *
 * Test Fixture
 *
 *****************************/
class TestComputationGraph : public testing::Test
{
protected:
	/**
	 * @brief Checks whether all of the value-updating and derivative-computing operations have been run.
	 *
	 */
	void _checkBackPropagation(
		const BackPropagationConfig& config,
		const std::map<mlCoreTests::OperatorStats::WrapperLogChannel, std::vector<std::string>>& expectedLogs)
	{
		for(const auto& [logChannel, logs] : expectedLogs)
		{
			const auto& collectedLogs = config.operatorStats->getLogs(logChannel);

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
	 * @param config Configuration of the back-propagation.
	 */
	void _performBackPropagation(const BackPropagationConfig& config)
	{
		using namespace autoDiff;

		const auto nodes = flattenTree(config.tree);

		_graph = std::make_unique<ComputationGraph>(ComputationGraphConfig{.useMultithreading = true});

		_graph->setRoot(config.tree);
		_graph->setDifferentiableNodes(config.trainableWeights);

		_graph->forwardPass();
		_graph->computeGradients(config.tree);
	}

protected:
	std::unique_ptr<autoDiff::ComputationGraph> _graph{nullptr};
};

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

	auto [tree, stats] = mlCoreTests::constructTree(treeConstructionConfig);

	const BackPropagationConfig config{.tree = tree,
									   .trainableWeights =
										   getNodesByNames(tree, {"L1W", "L1B", "L2W", "L2B", "L3W", "L3B"}),
									   .operatorStats = stats};

	_performBackPropagation(config);

	const std::map<mlCoreTests::OperatorStats::WrapperLogChannel, std::vector<std::string>> expectedLogs{
		{mlCoreTests::OperatorStats::WrapperLogChannel::UPDATE_VALUE,
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
		{mlCoreTests::OperatorStats::WrapperLogChannel::COMPUTE_DERIVATIVE,
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

	_checkBackPropagation(config, expectedLogs);
}

} // namespace
