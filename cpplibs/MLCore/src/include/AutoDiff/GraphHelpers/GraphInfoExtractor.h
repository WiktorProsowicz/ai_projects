#ifndef MLCORE_INCLUDE_AUTODIFF_GRAPHHELPERS_GRAPHINFOEXTRACTOR_H
#define MLCORE_INCLUDE_AUTODIFF_GRAPHHELPERS_GRAPHINFOEXTRACTOR_H

#include <map>

#include "AutoDiff/GraphNodes.hpp"

namespace autoDiff::detail
{
/**
 * @brief Wraps a part of computation graph and extracts information about it.
 */
class GraphInfoExtractor
{
public:
	GraphInfoExtractor() = delete;

	/**
	 * @brief Constructs the graph info extractor.
	 *
	 * @param root The root of the graph spanned by the extractor.
	 */
	explicit GraphInfoExtractor(NodePtr root)
		: _root(std::move(root))
		, _subtreeClasses(_composeSubtreeClasses())
	{}

	GraphInfoExtractor(const GraphInfoExtractor&) = default;
	GraphInfoExtractor(GraphInfoExtractor&&) = default;
	GraphInfoExtractor& operator=(const GraphInfoExtractor&) = default;
	GraphInfoExtractor& operator=(GraphInfoExtractor&&) = default;

	~GraphInfoExtractor() = default;

	/**
	 * @brief Computes the entropy score for a given node with respect to its subtrees.
	 *
	 * The entropy score is a measure of the distribution of nodes between the subtrees starting at the
	 * `node`. The score is close to zero for well-organized trees (i.e. trees for which the nodes are not too
	 * dispersed between multiple subtrees).
	 *
	 * @param node The node for which the entropy score should be computed.
	 */
	double getEntropyScore(const NodePtr& node) const;

	/**
	 * @brief Returns the size of a subtree starting from a given node.
	 *
	 */
	uint16_t getTreeSize(const NodePtr& node) const;

	/**
	 * @brief Returns a set of nodes that have entropy score above the given threshold.
	 *
	 * @param threshold The minimal entropy score for a node to be included in the result.
	 */
	std::vector<NodePtr> getNodesAboveEntropyThreshold(double threshold) const;

	/**
	 * @brief Returns the maximal number of subtrees encountered among the graph nodes.
	 *
	 */
	size_t getMaximalNumberOfSubtrees() const;

private:
	/// Creates a map containing nodes and sizes of their subtrees. The size is a number
	/// operators present in the subtree.
	std::map<NodePtr, std::vector<uint16_t>> _composeSubtreeClasses() const;

	NodePtr _root;
	std::map<NodePtr, std::vector<uint16_t>> _subtreeClasses;
};
} // namespace autoDiff::detail

#endif
