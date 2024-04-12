#include "AutoDiff/GraphHelpers/GraphInfoExtractor.h"

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

							   return curr - classProbability * std::log2(classProbability);
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

			for(const auto& input : castedOp->getInputs())
			{
				classSizes.push_back(getClassesForNode(input));
			}

			collectedClasses[node] = classSizes;

			return std::accumulate(classSizes.cbegin(), classSizes.cend(), uint16_t{0});
		}

		return 0;
	};

	getClassesForNode(_root);

	return collectedClasses;
}

uint16_t GraphInfoExtractor::getTreeSize(const NodePtr& node) const
{
	return std::accumulate(_subtreeClasses.at(node).cbegin(), _subtreeClasses.at(node).cend(), 1);
}

std::vector<NodePtr> GraphInfoExtractor::getNodesAboveEntropyThreshold(double threshold) const
{
	if(threshold < 0.0 || threshold > 1.0)
	{
		LOG_WARN("AutoDiff::GraphInfoExtractor", "Threshold must be in range [0, 1]");
		return {};
	}

	std::vector<NodePtr> nodesToProcess;

	for(const auto& node : _subtreeClasses |
							   std::views::filter([this, &threshold](const auto& item)
												  { return getEntropyScore(item.first) > threshold; }) |
							   std::views::keys)
	{
		nodesToProcess.emplace_back(node);
	}

	return nodesToProcess;
}
} // namespace autoDiff::detail