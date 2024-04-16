#include "AutoDiff/GraphHelpers/BackwardPassContext.h"

namespace autoDiff::detail
{
BackwardPassContext::BackwardPassContext(const BackwardPassParams& params)
	: _params(params)
	, _graphInfoExtractor(params.root)
	, _nodesForMultithreadedProcessing(_graphInfoExtractor.getNodesAboveEntropyThreshold(_entropyThreshold))
{}

void BackwardPassContext::run()
{
	if(!_params.useMultithreading)
	{
		_runBackwardPass(_params.root, 1.0);
		return;
	}

	_initThreadPool();

	_addEntryPoint(_params.root, 1.0);

	while(!_isEntryPointsQueueEmpty())
	{
		std::unique_lock lock(_entryPointsQueueMutex);

		while(!_entryPointsQueue.empty())
		{
			auto entryPoint = _entryPointsQueue.front();
			_entryPointsQueue.pop();

			_threadPool->addJob(
				[this, entryPoint]()
				{
					++_activeTasksCounter;
					_processFromEntryPoint(entryPoint);
					--_activeTasksCounter;

					_finishedTaskCv.notify_one();
				});
		}

		_finishedTaskCv.wait(lock, [this] { return _entryPointsQueue.empty(); });
	}

	_outerDerivatives.clear();
	_threadPool.reset();
}

void BackwardPassContext::_runBackwardPass(const NodePtr& node, const mlCore::Tensor& outerDerivative)
{
	_tryStoreDerivative(node, outerDerivative);

	if(const auto castedOp = std::dynamic_pointer_cast<Operator>(node))
	{
		const auto derivatives = castedOp->computeDerivative(outerDerivative);

		if(derivatives.size() != castedOp->getInputs().size())
		{
			LOG_ERROR("AutoDiff::BackwardPassContext",
					  "Encountered number of derivatives different than number of inputs.");
		}

		for(size_t inputIdx = 0; inputIdx < castedOp->getInputs().size(); ++inputIdx)
		{
			_runBackwardPass(castedOp->getInputs()[inputIdx], derivatives[inputIdx]);
		}
	}
}

void BackwardPassContext::_addEntryPoint(const NodePtr& node, mlCore::Tensor outerDerivative)
{
	const auto addedDerivativeIt = _registerOuterDerivative(std::move(outerDerivative));

	{
		std::unique_lock lock(_entryPointsQueueMutex);
		_entryPointsQueue.push({std::ref(*addedDerivativeIt), node});
	}
}

bool BackwardPassContext::_isEntryPointsQueueEmpty()
{
	std::shared_lock lock(_entryPointsQueueMutex);
	return _entryPointsQueue.empty();
}

void BackwardPassContext::_tryStoreDerivative(const NodePtr& node, const mlCore::Tensor& derivative)
{
	if(_params.differentiableNodes.contains(node))
	{
		std::unique_lock lock(_gradientsMutex);

		if(_params.gradients.find(node) == _params.gradients.end())
		{
			_params.gradients.emplace(node, derivative);
		}
		else
		{
			_params.gradients[node] += derivative;
		}
	}
}

void BackwardPassContext::_processFromEntryPoint(const PropagationEntryPoint& entryPoint)
{
	const auto& outerDerivative = entryPoint.outerDerivative.get();

	_tryStoreDerivative(entryPoint.rootNode, outerDerivative);

	if(const auto castedOp = std::dynamic_pointer_cast<Operator>(entryPoint.rootNode))
	{
		auto derivatives = castedOp->computeDerivative(outerDerivative);

		if(derivatives.size() != castedOp->getInputs().size())
		{
			LOG_ERROR("AutoDiff::BackwardPassContext",
					  "Encountered number of derivatives different than number of inputs.");
		}

		if(std::find(_nodesForMultithreadedProcessing.cbegin(),
					 _nodesForMultithreadedProcessing.cend(),
					 castedOp) != _nodesForMultithreadedProcessing.cend())
		{

			for(size_t inputIdx = 0; inputIdx < castedOp->getInputs().size(); ++inputIdx)
			{
				_addEntryPoint(castedOp->getInputs()[inputIdx], std::move(derivatives[inputIdx]));
			}
		}
		else
		{
			for(size_t inputIdx = 0; inputIdx < castedOp->getInputs().size(); ++inputIdx)
			{
				_processFromEntryPoint({std::ref(derivatives[inputIdx]), castedOp->getInputs()[inputIdx]});
			}
		}
	}
}

BackwardPassContext::TensorsStorage::iterator
BackwardPassContext::_registerOuterDerivative(mlCore::Tensor outerDerivative)
{
	std::unique_lock lock(_outerDerivativesMutex);
	return _outerDerivatives.emplace(std::move(outerDerivative)).first;
}

void BackwardPassContext::_initThreadPool()
{
	const auto maxSubtrees = _graphInfoExtractor.getMaximalNumberOfSubtrees();
	const size_t maxThreads = std::thread::hardware_concurrency() / 2;

	_threadPool = std::make_unique<utilities::ThreadPool>(std::min(maxSubtrees, maxThreads));
}
} // namespace autoDiff::detail
