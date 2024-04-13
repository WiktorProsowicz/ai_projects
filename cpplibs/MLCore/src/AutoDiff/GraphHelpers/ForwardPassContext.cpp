#include "AutoDiff/GraphHelpers/ForwardPassContext.h"

#include <ranges>
#include <shared_mutex>

namespace autoDiff::detail
{
ForwardPassContext::ForwardPassContext(bool useMultithreading, NodePtr root)
	: _useMultithreading(useMultithreading)
	, _root(std::move(root))
	, _graphInfoExtractor(_root)
{
	_nodesToProcess = _graphInfoExtractor.getNodesAboveEntropyThreshold(_entropyThreshold);

	std::sort(_nodesToProcess.begin(),
			  _nodesToProcess.end(),
			  [this](const auto& lhs, const auto& rhs)
			  { return _graphInfoExtractor.getTreeSize(lhs) < _graphInfoExtractor.getTreeSize(rhs); });
}

void ForwardPassContext::run()
{
	_visitedNodes.clear();

	if(!_useMultithreading)
	{
		_updateSubtree(_root);
		return;
	}

	_initThreadPool();

	std::for_each(_nodesToProcess.cbegin(),
				  _nodesToProcess.cend(),
				  [this](const auto& node) { _runInParallelFromNode(node); });

	_updateSubtree(_root);

	_threadPool.reset();
}

void ForwardPassContext::_initThreadPool()
{
	const auto maxSubtrees = _graphInfoExtractor.getMaximalNumberOfSubtrees();
	const size_t maxThreads = std::thread::hardware_concurrency() / 2;

	_threadPool = std::make_unique<utilities::ThreadPool>(std::min(maxSubtrees, maxThreads));
}

void ForwardPassContext::_runInParallelFromNode(const NodePtr& node)
{
	if(const auto castedOp = std::dynamic_pointer_cast<Operator>(node))
	{
		for(const auto& input : castedOp->getInputs())
		{
			_threadPool->addJob(
				[this, &input]()
				{
					_updateSubtree(input);
					_cv.notify_one();
				});
		}

		{
			std::unique_lock<std::shared_mutex> lock{_visitedNodesMutex};

			_cv.wait(lock,
					 [this, &castedOp]
					 {
						 return std::all_of(castedOp->getInputs().cbegin(),
											castedOp->getInputs().cend(),
											[this](const auto& input)
											{ return _visitedNodes.find(input) != _visitedNodes.end(); });
					 });
		}
	}
}

void ForwardPassContext::_markVisited(const NodePtr& node)
{
	_visitedNodes.insert(node);
}

bool ForwardPassContext::_isVisited(const NodePtr& node) const
{
	return _visitedNodes.find(node) != _visitedNodes.end();
}

void ForwardPassContext::_updateSubtree(const NodePtr& node)
{
	{
		std::unique_lock<std::shared_mutex> lock{_visitedNodesMutex};

		if(_isVisited(node))
		{
			return;
		}

		_markVisited(node);
	}

	if(const auto castedOp = std::dynamic_pointer_cast<Operator>(node))
	{
		for(const auto& input : castedOp->getInputs())
		{
			_updateSubtree(input);
		}

		castedOp->updateValue();
	}
}

std::map<NodePtr, std::vector<uint16_t>> ForwardPassContext::_composeSubtreeClasses() const
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

			if(castedOp->getInputs().size() > 1)
			{
				collectedClasses[node] = classSizes;
			}

			return std::accumulate(classSizes.cbegin(), classSizes.cend(), uint16_t{0});
		}

		return 0;
	};

	getClassesForNode(_root);

	return collectedClasses;
}
} // namespace autoDiff::detail