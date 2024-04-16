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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ComputationGraphScenarios.hpp"
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
	bool multiThreaded;
};

/*****************************
 *
 * Common functions
 *
 *****************************/

std::set<autoDiff::NodePtr> getNodesByNames(const std::map<std::string, autoDiff::NodePtr> nodesMap,
											const std::set<std::string>& nodeNames)
{

	std::set<autoDiff::NodePtr> nodes;

	std::transform(nodeNames.cbegin(),
				   nodeNames.cend(),
				   std::inserter(nodes, nodes.begin()),
				   [&nodesMap](const auto& name) { return nodesMap.at(name); });

	return nodes;
}

/// Verifies that vector B consists of all elements from A. Vector B is allowed to contains additional
/// elements.
template <typename T>
::testing::AssertionResult vectorBIsReducibleToA(const std::vector<T>& vectorA, const std::vector<T>& vectorB)
{
	std::vector<std::reference_wrapper<const T>> vectorARefs;
	std::vector<std::reference_wrapper<const T>> vectorBRefs;

	std::for_each(vectorA.cbegin(),
				  vectorA.cend(),
				  [&vectorARefs](const auto& element) { vectorARefs.emplace_back(element); });

	std::for_each(vectorB.cbegin(),
				  vectorB.cend(),
				  [&vectorBRefs](const auto& element) { vectorBRefs.emplace_back(element); });

	std::vector<std::reference_wrapper<const T>> missingElements;

	while(!vectorARefs.empty())
	{
		const auto& soughtElement = vectorARefs.back();

		const auto foundElement = std::find_if(vectorBRefs.begin(),
											   vectorBRefs.end(),
											   [&soughtElement](const auto& element)
											   { return element.get() == soughtElement.get(); });

		if(foundElement == vectorBRefs.end())
		{
			missingElements.emplace_back(soughtElement);
		}

		vectorARefs.pop_back();
		vectorBRefs.erase(foundElement);
	}

	if(!missingElements.empty())
	{
		std::stringstream ss;

		ss << "Missing elements: ";

		for(const auto& element : missingElements)
		{
			ss << element.get() << ", ";
		}

		return ::testing::AssertionFailure() << ss.str();
	}

	return ::testing::AssertionSuccess();
}

/*****************************
 *
 * Test Fixture
 *
 *****************************/
class TestComputationGraph : public testing::Test
{
public:
	/**
	 * @brief Checks whether all of the value-updating and derivative-computing operations have been run.
	 *
	 */
	static void assertLogsAreAsExpected(
		const BackPropagationConfig& config,
		const std::map<mlCoreTests::OperatorStats::WrapperLogChannel, std::vector<std::string>>& expectedLogs)
	{
		for(const auto& [logChannel, logs] : expectedLogs)
		{
			const auto& collectedLogs = config.operatorStats->getLogs(logChannel);

			ASSERT_TRUE(vectorBIsReducibleToA<std::string>(logs, collectedLogs));
		}
	}

	void assertGradientsAreAvailable(const BackPropagationConfig& config) const
	{
		ASSERT_THAT(config.trainableWeights,
					::testing::Each(::testing::Truly([this](const auto& weight)
													 { return _graph->hasGradient(weight); })));
	};

protected:
	/**
	 * @brief Simulates computation graph forward pass and gradients updates.
	 *
	 * @param config Configuration of the back-propagation.
	 */
	void _performBackPropagation(const BackPropagationConfig& config)
	{
		using namespace autoDiff;

		_graph = std::make_unique<ComputationGraph>(
			ComputationGraphConfig{.useMultithreading = config.multiThreaded});

		_graph->setRoot(config.tree);
		_graph->setDifferentiableNodes(config.trainableWeights);

		_graph->forwardPass();
		_graph->computeGradients(config.tree);
	}

protected:
	std::unique_ptr<autoDiff::ComputationGraph> _graph{nullptr};
};

TEST_F(TestComputationGraph, CollectProperLogsForOneRootTreeMultithreaded)
{
	auto [nodes, stats] = mlCoreTests::constructTree(mlCoreTests::treeConfigOneRoot);

	const BackPropagationConfig config{.tree = nodes.at("OutputLayer"),
									   .trainableWeights =
										   getNodesByNames(nodes, {"L1W", "L1B", "L2W", "L2B", "L3W", "L3B"}),
									   .operatorStats = stats,
									   .multiThreaded = true};

	_performBackPropagation(config);

	assertLogsAreAsExpected(config, mlCoreTests::expectedLogsOneRoot);
	assertGradientsAreAvailable(config);
}

TEST_F(TestComputationGraph, CollectProperLogsForOneRootTreeSinglethreaded)
{
	auto [nodes, stats] = mlCoreTests::constructTree(mlCoreTests::treeConfigOneRoot);

	const BackPropagationConfig config{.tree = nodes.at("OutputLayer"),
									   .trainableWeights =
										   getNodesByNames(nodes, {"L1W", "L1B", "L2W", "L2B", "L3W", "L3B"}),
									   .operatorStats = stats,
									   .multiThreaded = false};

	_performBackPropagation(config);

	assertLogsAreAsExpected(config, mlCoreTests::expectedLogsOneRoot);
	assertGradientsAreAvailable(config);
}

TEST_F(TestComputationGraph, CollectProperLogsForMultipleRootTreeMultithreaded)
{
	auto [nodes, stats] = mlCoreTests::constructTree(mlCoreTests::treeConfigXShape);

	const BackPropagationConfig config{
		.tree = nodes["OutputLayer"],
		.trainableWeights = getNodesByNames(
			nodes, {"LeftInput/L1/W",	"LeftInput/L1/B",	"LeftInput/L2/W",	"LeftInput/L2/B",
					"LeftInput/L3/W",	"LeftInput/L3/B",	"RightInput/L1/W",	"RightInput/L1/B",
					"RightInput/L2/W",	"RightInput/L2/B",	"RightInput/L3/W",	"RightInput/L3/B",
					"LeftOutput/L1/W",	"LeftOutput/L1/B",	"LeftOutput/L2/W",	"LeftOutput/L2/B",
					"LeftOutput/L3/W",	"LeftOutput/L3/B",	"RightOutput/L1/W", "RightOutput/L1/B",
					"RightOutput/L2/W", "RightOutput/L2/B", "RightOutput/L3/W", "RightOutput/L3/B"}),
		.operatorStats = stats,
		.multiThreaded = true};

	_performBackPropagation(config);

	assertLogsAreAsExpected(config, mlCoreTests::expectedLogsXShape);
	assertGradientsAreAvailable(config);
}
} // namespace
