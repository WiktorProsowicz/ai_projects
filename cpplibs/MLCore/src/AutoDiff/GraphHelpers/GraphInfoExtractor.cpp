#include "AutoDiff/GraphHelpers/GraphInfoExtractor.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <ranges>

#include <LoggingLib/LoggingLib.hpp>

namespace autoDiff::detail
{
double GraphInfoExtractor::getEntropyScore(const NodePtr& node) const
{
	const double classesSum = getTreeSize(node) - 1;

	return std::accumulate(_subtreeClasses.at(node).cbegin(),
						   _subtreeClasses.at(node).cend(),
						   0.0,
						   [&classesSum](const auto curr, const auto subtreeSize)
						   {
							   const auto classProbability = static_cast<double>(subtreeSize) / classesSum;

							   return curr - (classProbability * std::log2(classProbability));
						   });
}

std::map<NodePtr, std::vector<uint16_t>> GraphInfoExtractor::_composeSubtreeClasses() const
{
	std::map<NodePtr, std::vector<uint16_t>> collectedClasses;

	// Fills the classes map and returns the size of a subtree starting from a given node.
	std::function<uint16_t(const NodePtr&)> getClassesForNode;
	getClassesForNode = [&collectedClasses, &getClassesForNode](const NodePtr& node) -> uint16_t
	{
		if(const auto castedOp = std::dynamic_pointer_cast<Operator>(node))
		{
			std::vector<uint16_t> classSizes;
			classSizes.reserve(castedOp->getInputs().size());

			std::ranges::transform(castedOp->getInputs(),
								   std::back_inserter(classSizes),
								   [&getClassesForNode](const NodePtr& opInput)
								   { return getClassesForNode(opInput); });

			const auto subtreeSize = std::accumulate(classSizes.cbegin(), classSizes.cend(), uint16_t{0});

			collectedClasses.emplace(node, std::move(classSizes));

			return subtreeSize;
		}

		return 0;
	};

	getClassesForNode(_root);

	return collectedClasses;
}

size_t GraphInfoExtractor::getMaximalNumberOfSubtrees() const
{
	const auto elementWithMaxSubtrees = std::ranges::max_element(
		_subtreeClasses.cbegin(),
		_subtreeClasses.cend(),
		[](const auto& lhs, const auto& rhs) { return lhs.second.size() < rhs.second.size(); });

	return elementWithMaxSubtrees->second.size();
}

uint16_t GraphInfoExtractor::getTreeSize(const NodePtr& node) const
{
	return std::accumulate(_subtreeClasses.at(node).cbegin(), _subtreeClasses.at(node).cend(), uint16_t{1});
}

std::vector<NodePtr> GraphInfoExtractor::getNodesAboveEntropyThreshold(double threshold) const
{
	if(threshold < 0.0 || threshold > 1.0)
	{
		LOG_WARN("AutoDiff::GraphInfoExtractor", "Threshold must be in range [0, 1]");
		return {};
	}

	std::vector<NodePtr> chosenNodes;

	std::ranges::copy(_subtreeClasses |
						  std::views::filter([this, &threshold](const auto& item)
											 { return getEntropyScore(item.first) > threshold; }) |
						  std::views::keys,
					  std::back_inserter(chosenNodes));

	return chosenNodes;
}

} // namespace autoDiff::detail
