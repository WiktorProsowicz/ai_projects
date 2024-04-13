#ifndef MLCORE_SRC_INCLUDE_AUTODIFF_GRAPHHELPERS_FORWARDPASSCONTEXT_H
#define MLCORE_SRC_INCLUDE_AUTODIFF_GRAPHHELPERS_FORWARDPASSCONTEXT_H

#include <condition_variable>
#include <set>

#include <Utilities/ThreadPool.h>

#include "AutoDiff/GraphHelpers/GraphInfoExtractor.h"
#include "AutoDiff/GraphNodes.hpp"

namespace autoDiff::detail
{
/**
 * @brief Performs forward pass on the graph.
 *
 */
class ForwardPassContext
{
public:
	ForwardPassContext() = delete;

	/**
	 * @brief Constructs the forward pass context.
	 *
	 * @param useMultithreading Tells if the algorithm should analyze the graph and run the forward pass in
	 * parallel.
	 * @param root The root of the graph for which the forward pass should be performed.
	 */
	ForwardPassContext(bool useMultithreading, NodePtr root);

	ForwardPassContext(const ForwardPassContext&) = delete;
	ForwardPassContext(ForwardPassContext&&) = delete;
	ForwardPassContext& operator=(const ForwardPassContext&) = delete;
	ForwardPassContext& operator=(ForwardPassContext&&) = delete;

	~ForwardPassContext() = default;

	/**
	 * @brief Traverses the graph starting from the deepest operators found from the root perspective.
	 *
	 * If the forward pass is to be performed in a multithreaded manner, an additional algorithm shall be run
	 * to determine the entropy score for each operator in the graph that divides the graph into subgraphs
	 * that can be computed in parallel. The computed score allows to decide, whether a thread pool should be
	 * used and, if so, how many threads should be assigned to it.
	 */
	void run();

private:
	/// Tells how big should be the entropy score computed from a given node's perspective to run forward pass
	/// in parallel starting at it.
	static constexpr double _entropyThreshold = 0.7;

	/// Runs the forward pass using the internal thread pool starting from the provided node. The function
	/// terminated after the internal thread pool updates all inputs of the `node`.
	void _runInParallelFromNode(const NodePtr& node);

	/// Tells if a given node has been visited during the forward pass.
	bool _isVisited(const NodePtr& node) const;

	/// Marks a given node as visited during the forward pass.
	void _markVisited(const NodePtr& node);

	/// Updates the subtree starting at a given node recursively. The algorithm stops at node that has been
	/// marked as visited.
	void _updateSubtree(const NodePtr& node);

	/// Assigns the internal thread pool according to provided configuration.
	void _initThreadPool();

	/// Creates a map containing nodes and sizes of their subtrees. The size is a number
	/// operators present in the subtree. Only the nodes having at least two subtrees are taken into account.
	std::map<NodePtr, std::vector<uint16_t>> _composeSubtreeClasses() const;

	bool _useMultithreading;
	NodePtr _root;
	/// Used to run the forward-pass tasks in parallel.
	std::unique_ptr<utilities::ThreadPool> _threadPool{};
	std::condition_variable_any _cv{};
	/// Contains nodes that have been visited during the forward pass.
	std::set<NodePtr> _visitedNodes{};
	std::shared_mutex _visitedNodesMutex{};
	/// Yields necessary info about the spanned graph.
	GraphInfoExtractor _graphInfoExtractor;
	/// Contains nodes fot that the forward pass should be run in parallel.
	std::vector<NodePtr> _nodesToProcess{};
};
} // namespace autoDiff::detail

#endif