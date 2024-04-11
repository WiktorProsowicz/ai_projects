#include "AutoDiff/GraphHelpers/ForwardPassContext.h"

#include <ranges>
#include <shared_mutex>

namespace autoDiff::detail
{
void ForwardPassContext::run()
{
	if(!_useMultithreading)
	{
		_updateSubtree(_root);
		return;
	}

	const auto subtreeClasses = _composeSubtreeClasses();

	const auto computeEntropy = [&subtreeClasses](const NodePtr& node)
	{
		const auto classesSum =
			std::accumulate(subtreeClasses.at(node).cbegin(), subtreeClasses.at(node).cend(), 0.0);

		return std::accumulate(subtreeClasses.at(node).cbegin(),
							   subtreeClasses.at(node).cend(),
							   0.0,
							   [&classesSum](const auto curr, const auto subtreeSize)
							   {
								   const auto classProbability =
									   static_cast<double>(subtreeSize) / classesSum;

								   return curr - classProbability * std::log2(classProbability);
							   });
	};

	_initThreadPool(subtreeClasses);

	std::vector<NodePtr> nodesToProcess;

	for(const auto& node :
		subtreeClasses |
			std::views::filter([&computeEntropy](const auto& item)
							   { return computeEntropy(item.first) > _entropyThreshold; }) |
			std::views::keys)
	{
		nodesToProcess.emplace_back(node);
	}

	std::sort(nodesToProcess.begin(),
			  nodesToProcess.end(),
			  [&subtreeClasses](const auto& lhs, const auto& rhs)
			  {
				  return std::accumulate(subtreeClasses.at(lhs).cbegin(), subtreeClasses.at(lhs).cend(), 0) >
						 std::accumulate(subtreeClasses.at(rhs).cbegin(), subtreeClasses.at(rhs).cend(), 0);
			  });

	std::for_each(nodesToProcess.cbegin(),
				  nodesToProcess.cend(),
				  [this](const auto& node) { _runInParallelFromNode(node); });

	_updateSubtree(_root);
}

void ForwardPassContext::_initThreadPool(const std::map<NodePtr, std::vector<uint16_t>>& subtreeClasses)
{
	const auto elementWithMaxSubtrees = std::max_element(subtreeClasses.cbegin(),
														 subtreeClasses.cend(),
														 [](const auto& lhs, const auto& rhs)
														 { return lhs.second.size() < rhs.second.size(); });

	const size_t maxThreads = std::thread::hardware_concurrency() / 2;

	_threadPool =
		std::make_unique<utilities::ThreadPool>(std::min(elementWithMaxSubtrees->second.size(), maxThreads));
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