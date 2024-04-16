#ifndef MLCORE_SRC_INCLUDE_AUTODIFF_GRAPHHELPERS_BACKWARDPASSCONTEXT_H
#define MLCORE_SRC_INCLUDE_AUTODIFF_GRAPHHELPERS_BACKWARDPASSCONTEXT_H

#include <condition_variable>
#include <queue>
#include <set>

#include <Utilities/ThreadPool.h>
#include <Utilities/ThreadSafeQueue.hpp>

#include "AutoDiff/GraphHelpers/GraphInfoExtractor.h"
#include "AutoDiff/GraphNodes.hpp"

namespace autoDiff::detail
{
/// @brief Compares two objects by their addresses.
template <typename T>
struct AddressComparator
{
	bool operator()(const T& lhs, const T& rhs) const
	{
		return std::addressof(lhs) < std::addressof(rhs);
	}
};

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
	/// Tells how big should be the entropy score computed from a given node's perspective to run forward pass
	/// in parallel starting at it.
	static constexpr double _entropyThreshold = 0.7;

	using TensorsStorage = std::set<mlCore::Tensor, AddressComparator<mlCore::Tensor>>;

	/// Contains information about a point from which back-propagation should be started.
	struct PropagationEntryPoint
	{
		std::reference_wrapper<const mlCore::Tensor> outerDerivative;
		NodePtr rootNode;
	};

	/// Assigns the internal thread pool according to provided configuration.
	void _initThreadPool();

	/// Adds a tensor to the outer derivatives storage so that it can be referenced by the propagation entry
	/// points.
	TensorsStorage::iterator _registerOuterDerivative(mlCore::Tensor outerDerivative);

	/// Tells whether the entry points queue is empty. The check is thread-safe.
	bool _isEntryPointsQueueEmpty();

	/// Stores derivative for a given node, if it present in the differential nodes set.
	void _tryStoreDerivative(const NodePtr& node, const mlCore::Tensor& derivative);

	/// Adds an entry point to the queue. The function is thread-safe.
	void _addEntryPoint(const NodePtr& node, mlCore::Tensor outerDerivative);

	/// Runs back-propagation from the provided entry point.
	void _processFromEntryPoint(const PropagationEntryPoint& entryPoint);

	/// Runs back-propagation starting from a given node using single thread.
	void _runBackwardPass(const NodePtr& node, const mlCore::Tensor& outerDerivative);

	BackwardPassParams _params;
	GraphInfoExtractor _graphInfoExtractor;
	std::condition_variable_any _finishedTaskCv{};
	/// Contains the nodes for which the gradients should be computed. The entry points shall be taken from it
	/// by the internal thread pool and processed.
	std::queue<PropagationEntryPoint> _entryPointsQueue{};
	std::shared_mutex _entryPointsQueueMutex{};
	std::unique_ptr<utilities::ThreadPool> _threadPool{};
	/// Contains the nodes for which the forward pass should be run in parallel.
	const std::vector<NodePtr> _nodesForMultithreadedProcessing{};
	/// Contains the nodes for which references are stored in the _entryPointsQueue.
	/// This set is cleared after a full backward pass is performed.
	TensorsStorage _outerDerivatives{};
	std::shared_mutex _outerDerivativesMutex{};
	std::shared_mutex _gradientsMutex{};
	std::atomic_uint16_t _activeTasksCounter = 0;
};
} // namespace autoDiff::detail

#endif
