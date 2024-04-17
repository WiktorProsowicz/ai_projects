#include "AutoDiff/GraphHelpers/GraphSerializer.h"

#include <ranges>

#include <fmt/core.h>

namespace autoDiff::detail
{
GraphSerializer::GraphSerializer(NodePtr root)
	: _root(root)
{
	_determineNodesLevels();
	_determineNodesConnections();
}

std::string GraphSerializer::serialize() const
{
	constexpr const char* mainStructure =
		R"(digraph G {{ bgcolor="darkslategrey"; {global_attributes} {node_clusters} {node_connections}}})";

	return fmt::format(mainStructure,
					   fmt::arg("global_attributes", _getGlobalGraphAttributes()),
					   fmt::arg("node_clusters", fmt::join(_serializeNodesClusters(), " ")),
					   fmt::arg("node_connections", fmt::join(_serializeNodesConnections(), " ")));
}

void GraphSerializer::_determineNodesLevels()
{
	std::map<NodePtr, std::set<size_t>> collectedDepths;

	std::function<void(const NodePtr&, size_t)> determineNodeDepth;

	determineNodeDepth = [&collectedDepths, &determineNodeDepth](const NodePtr& node, size_t depth)
	{
		collectedDepths[node].emplace(depth);

		if(const auto castedOp = std::dynamic_pointer_cast<Operator>(node))
		{
			for(const auto& input : castedOp->getInputs())
			{
				determineNodeDepth(input, depth + 1);
			}
		}
	};

	determineNodeDepth(_root, 0);

	std::map<NodePtr, size_t> finalDepths;
	std::transform(collectedDepths.cbegin(),
				   collectedDepths.cend(),
				   std::inserter(finalDepths, finalDepths.begin()),
				   [](const auto& pair) { return std::make_pair(pair.first, *pair.second.rbegin()); });

	const auto maximalDepth =
		std::max_element(finalDepths.cbegin(),
						 finalDepths.cend(),
						 [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; })
			->second;

	std::for_each(finalDepths.cbegin(),
				  finalDepths.cend(),
				  [this, &maximalDepth](const auto& pair)
				  { _nodesLevels[pair.first] = maximalDepth - pair.second; });
}

void GraphSerializer::_determineNodesConnections()
{
	std::function<void(const NodePtr&)> saveConnectionsForNode;

	saveConnectionsForNode = [this, &saveConnectionsForNode](const NodePtr& node)
	{
		if(const auto castedOp = std::dynamic_pointer_cast<Operator>(node))
		{

			for(const auto& input : castedOp->getInputs())
			{
				_nodesConnections[input].emplace(castedOp);

				saveConnectionsForNode(input);
			}
		}
	};

	saveConnectionsForNode(_root);
}

std::string GraphSerializer::_getNodeIdentifier(const NodePtr& node) const
{
	return fmt::format("_{}", std::hash<NodePtr>{}(node));
}

std::string GraphSerializer::_getGlobalGraphAttributes() const
{
	constexpr const char* globalAttributes = R"(ranksep={rank_sep}; size="{width},{height}";)";

	constexpr float defaultWidth = 10.0;

	return fmt::format(globalAttributes,
					   fmt::arg("rank_sep", _rankSpacing),
					   fmt::arg("height", _getGraphHeight()),
					   fmt::arg("width", defaultWidth));
}

std::vector<std::string> GraphSerializer::_serializeNodesClusters() const
{
	constexpr const char* clusterFormat = R"({{ rank=same; {nodes_definitions} }})";

	std::vector<std::string> stringifiedClusters;

	for(size_t level = 0; level <= _getMaxNodeLevel(); ++level)
	{
		std::vector<std::string> nodesDefinitions;

		for(const auto& node :
			_nodesLevels |
				std::ranges::views::filter([level](const auto& pair) { return pair.second == level; }) |
				std::ranges::views::keys)
		{
			nodesDefinitions.emplace_back(_getNodeDefinition(node));
		}

		stringifiedClusters.emplace_back(
			fmt::format(clusterFormat, fmt::arg("nodes_definitions", fmt::join(nodesDefinitions, " "))));
	}

	return stringifiedClusters;
}

std::vector<std::string> GraphSerializer::_serializeNodesConnections() const
{
	constexpr const char* connectionFormat = R"({parent} -> {child} [color="cyan4"];)";

	std::vector<std::string> stringifiedConnections;

	for(const auto& [parent, children] : _nodesConnections)
	{
		for(const auto& child : children)
		{
			stringifiedConnections.emplace_back(fmt::format(connectionFormat,
															fmt::arg("parent", _getNodeIdentifier(parent)),
															fmt::arg("child", _getNodeIdentifier(child))));
		}
	}

	return stringifiedConnections;
}

std::string GraphSerializer::_getNodeDefinition(const NodePtr& node) const
{
	constexpr const char* nodeFormat =
		R"({node_id} [label="{node_name}|{node_output_shape}"; color="{node_color}"; shape="record"; fontcolor="white"; style="bold"];)";

	if(const auto castedOp = std::dynamic_pointer_cast<Operator>(node))
	{
		return fmt::format(nodeFormat,
						   fmt::arg("node_id", _getNodeIdentifier(node)),
						   fmt::arg("node_name", castedOp->getName()),
						   fmt::arg("node_output_shape", mlCore::stringifyVector(castedOp->getOutputShape())),
						   fmt::arg("node_color", "honeydew2"));
	}

	if(const auto castedVar = std::dynamic_pointer_cast<Variable>(node))
	{
		return fmt::format(
			nodeFormat,
			fmt::arg("node_id", _getNodeIdentifier(node)),
			fmt::arg("node_name", castedVar->getName()),
			fmt::arg("node_output_shape", mlCore::stringifyVector(castedVar->getOutputShape())),
			fmt::arg("node_color", "darkgoldenrod3"));
	}

	if(const auto castedConst = std::dynamic_pointer_cast<Constant>(node))
	{
		return fmt::format(
			nodeFormat,
			fmt::arg("node_id", _getNodeIdentifier(node)),
			fmt::arg("node_name", castedConst->getName()),
			fmt::arg("node_output_shape", mlCore::stringifyVector(castedConst->getOutputShape())),
			fmt::arg("node_color", "green"));
	}

	if(const auto castedPlaceholder = std::dynamic_pointer_cast<Placeholder>(node))
	{
		return fmt::format(
			nodeFormat,
			fmt::arg("node_id", _getNodeIdentifier(node)),
			fmt::arg("node_name", castedPlaceholder->getName()),
			fmt::arg("node_output_shape", mlCore::stringifyVector(castedPlaceholder->getOutputShape())),
			fmt::arg("node_color", "dodgerblue2"));
	}

	return {};
}

size_t GraphSerializer::_getMaxNodeLevel() const
{
	if(_nodesLevels.empty())
	{
		return 0;
	}

	return std::max_element(_nodesLevels.cbegin(),
							_nodesLevels.cend(),
							[](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; })
		->second;
}
} // namespace autoDiff::detail