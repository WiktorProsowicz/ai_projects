#ifndef MLCORE_SRC_INCLUDE_AUTODIFF_GRAPHHELPERS_BACKWARDPASSCONTEXT_H
#define MLCORE_SRC_INCLUDE_AUTODIFF_GRAPHHELPERS_BACKWARDPASSCONTEXT_H

#include <set>

#include "AutoDiff/GraphNodes.hpp"

namespace autoDiff::detail
{
class BackwardPassContext
{
public:
	BackwardPassContext() = delete;

	/**
	 * @brief Constructs the backward-pass context.
	 *
	 * @param useMultithreading Tells whether the backward pass should be run in parallel.
	 * @param root The node from which the backward pass should be started.
	 * @param differentiableNodes A set of nodes for which the gradients should be stored.
	 * @param gradients A map in which the computed gradients should be stored.
	 */
	BackwardPassContext(const bool useMultithreading,
						NodePtr root,
						const std::set<NodePtr>& differentiableNodes,
						std::map<NodePtr, mlCore::Tensor>& gradients)
		: _useMultithreading(useMultithreading)
		, _root(std::move(root))
		, _differentiableNodes(differentiableNodes)
		, _gradients(gradients)
	{}

	BackwardPassContext(const BackwardPassContext&) = delete;
	BackwardPassContext(BackwardPassContext&&) = delete;
	BackwardPassContext& operator=(const BackwardPassContext&) = delete;
	BackwardPassContext& operator=(BackwardPassContext&&) = delete;

	~BackwardPassContext() = default;

private:
	bool _useMultithreading;
	NodePtr _root;

	const std::set<NodePtr>& _differentiableNodes;
	std::map<NodePtr, mlCore::Tensor>& _gradients;
};
} // namespace autoDiff::detail

#endif