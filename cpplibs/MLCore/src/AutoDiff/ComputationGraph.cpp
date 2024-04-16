#include "AutoDiff/ComputationGraph.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <LoggingLib/LoggingLib.hpp>

#include "AutoDiff/GraphHelpers/BackwardPassContext.h"
#include "AutoDiff/GraphHelpers/ForwardPassContext.h"
#include "AutoDiff/GraphNodes.hpp"
#include "MLCore/BasicTensor.h"

namespace autoDiff
{
namespace detail
{
/// @brief Contains backward contexts associated with nodes being set as their roots.
class BackwardPassContextsMap : public std::map<NodePtr, BackwardPassContext>
{
public:
	using std::map<NodePtr, BackwardPassContext>::map;
};
} // namespace detail

ComputationGraph::ComputationGraph(const ComputationGraphConfig& config)
	: _config(config)
	, _backwardPassContexts(std::make_unique<detail::BackwardPassContextsMap>())
{}

// NOLINTBEGIN(modernize-use-equals-default)

ComputationGraph::~ComputationGraph() {}

// NOLINTEND(modernize-use-equals-default)

bool ComputationGraph::hasGradient(const NodePtr& node) const
{
	return _gradients.find(node) != _gradients.end();
}

const mlCore::Tensor& ComputationGraph::getGradient(const NodePtr& node) const
{
	return _gradients.at(node);
}

void ComputationGraph::forwardPass()
{
	if(!_root)
	{
		LOG_ERROR("AutoDiff::ComputationGraph", "Root node is not set!");
		return;
	}

	_forwardPassContext->run();
}

void ComputationGraph::computeGradients(const NodePtr& backPropRoot)
{
	if(!_root)
	{
		LOG_ERROR("AutoDiff::ComputationGraph", "Root node is not set!");
		return;
	}

	if(!_backwardPassContexts->contains(backPropRoot))
	{
		const auto backwardPassParams =
			detail::BackwardPassParams{.useMultithreading = _config.useMultithreading,
									   .root = backPropRoot,
									   .differentiableNodes = _differentiableNodes,
									   .gradients = _gradients};

		_backwardPassContexts->emplace(backPropRoot, backwardPassParams);
	}

	_backwardPassContexts->at(backPropRoot).run();
}

void ComputationGraph::setRoot(const NodePtr& root)
{
	if(root != _root)
	{
		_forwardPassContext = std::make_unique<detail::ForwardPassContext>(_config.useMultithreading, root);

		_root = root;
	}
}

void ComputationGraph::setDifferentiableNodes(const std::set<NodePtr>& nodes)
{
	_differentiableNodes = nodes;
}
} // namespace autoDiff
