#ifndef MLCORE_INCLUDE_AUTODIFF_GRAPHHELPERS_GRAPHSERIALIZER_H
#define MLCORE_INCLUDE_AUTODIFF_GRAPHHELPERS_GRAPHSERIALIZER_H

#include <set>

#include "AutoDiff/GraphNodes.hpp"

namespace autoDiff::detail
{
/**
 * @brief Serializes the computation graph into the DOT format.
 *
 * @details The nodes spanned by the graph are given labels informing of their type, output shape, name
 * etc. The output graph is a directed, acyclic graph presenting connections between nodes. The nodes are
 * aligned according to their depth in the graph.
 */
class GraphSerializer
{
public:
	GraphSerializer() = delete;

	/**
	 * @brief Constructs the serializer assigning the spanned part of the graph to it.
	 *
	 * @param root The root node of the part of the graph to be serialized.
	 */
	GraphSerializer(NodePtr root);

	GraphSerializer(const GraphSerializer&) = delete;
	GraphSerializer& operator=(const GraphSerializer&) = delete;

	GraphSerializer(GraphSerializer&&) = delete;
	GraphSerializer& operator=(GraphSerializer&&) = delete;

	~GraphSerializer() = default;

	/**
	 * @brief Serializes the spanned part of the graph into the DOT format.
	 *
	 */
	std::string serialize() const;

private:
	/// The distance between various levels in the graph,
	constexpr static float _rankSpacing = 0.5;

	/// Stores the level at which the nodes should be placed.
	void _determineNodesLevels();

	/// Stores the connections between operators and their inputs.
	void _determineNodesConnections();

	/// Creates a string id representing the given node.
	std::string _getNodeIdentifier(const NodePtr& node) const;

	/// Returns the attributes of the DOT graph structure.
	std::string _getGlobalGraphAttributes() const;

	/// Returns the stringified forms of nodes at each level.
	std::vector<std::string> _serializeNodesClusters() const;

	/// Returns the stringified forms of connections between nodes.
	std::vector<std::string> _serializeNodesConnections() const;

	/// Returns stringified version of the given node, including its attributes.
	std::string _getNodeDefinition(const NodePtr& node) const;

	size_t _getMaxNodeLevel() const;

	float _getGraphHeight() const
	{
		return static_cast<float>(_getMaxNodeLevel() + 1) * _rankSpacing;
	}

	NodePtr _root;
	std::map<NodePtr, std::set<NodePtr>> _nodesConnections{};
	std::map<NodePtr, size_t> _nodesLevels{};
};
} // namespace autoDiff::detail

#endif
