#ifndef MLCORE_SRC_INCLUDE_AUTODIFF_GRAPHHELPERS_BACKWARDPASSCONTEXT_H
#define MLCORE_SRC_INCLUDE_AUTODIFF_GRAPHHELPERS_BACKWARDPASSCONTEXT_H

#include <condition_variable>
#include <set>

#include <Utilities/ThreadPool.h>
#include <Utilities/ThreadSafeQueue.hpp>

#include "AutoDiff/GraphHelpers/GraphInfoExtractor.h"
#include "AutoDiff/GraphNodes.hpp"

namespace autoDiff::detail
{
/**
 * @brief Contains parameters used by the backward pass context.
 */
struct BackwardPassParams
{
	/// Tells whether the backward pass should be run in parallel.
	bool useMultithreading;
	/// The node from which the backward pass should be started.
	NodePtr root;
	/// A set of nodes for which the gradients should be stored.
	const std::set<NodePtr>& differentiableNodes;
	/// A map in which the computed gradients should be stored.
	std::map<NodePtr, mlCore::Tensor>& gradients;
};

/**
 * @brief Contains parameters and algorithms used to perform back-propagation.
 *
 */
class BackwardPassContext

{
public:
	BackwardPassContext() = delete;

	/**
	 * @brief Constructs the backward-pass context.
	 *
	 * @param params Parameters of the backward pass.
	 */
	BackwardPassContext(const BackwardPassParams& params);

	BackwardPassContext(const BackwardPassContext&) = delete;
	BackwardPassContext(BackwardPassContext&&) = delete;
	BackwardPassContext& operator=(const BackwardPassContext&) = delete;
	BackwardPassContext& operator=(BackwardPassContext&&) = delete;

	~BackwardPassContext() = default;

	/**
	 * @brief Performs the back-propagation algorithm.
	 *
	 * The algorithm computes the gradients of the context's root node with respect to all nodes in the graph.
	 * Gradients for nodes set as context's differentiable nodes are stored in the context's gradients map. If
	 * the multithreading is supported, the algorithm will run in parallel for nodes for which a sufficient
	 * entropy score is computed. The entropy score of a particular node is computed as the distribution of
	 * child nodes between its subtrees.
	 *
	 */
	void run();

private:
	/// Contains information about a point from which back-propagation should be started.
	struct PropagationEntryPoint
	{
		std::reference_wrapper<const mlCore::Tensor> outerDerivative;
		NodePtr rootNode;
	};

	BackwardPassParams _params;
	GraphInfoExtractor _graphInfoExtractor;
	std::condition_variable _finishedTaskCv{};
	utilities::ThreadSafeQueue<PropagationEntryPoint> _entryPointsQueue{};
	std::unique_ptr<utilities::ThreadPool> _threadPool{};
};
} // namespace autoDiff::detail

#endif