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

#include "AutoDiff/GraphHelpers/ForwardPassContext.h"
#include "AutoDiff/GraphNodes.hpp"
#include "MLCore/BasicTensor.h"

namespace autoDiff
{
ComputationGraph::ComputationGraph(const ComputationGraphConfig& config)
	: _config(config)
{}

bool ComputationGraph::hasGradient(const NodePtr& node) const
{
	return _gradients.find(node) == _gradients.end();
}

const mlCore::Tensor& ComputationGraph::getGradient(const NodePtr& node) const
{
	return _gradients.at(node);
}

void ComputationGraph::forwardPass()
{
	if(!_root)
	{
		LOG_ERROR("ComputationGraph", "Root node is not set!");
		return;
	}

	detail::ForwardPassContext context(_config.useMultithreading, _root);

	context.run();
}

void ComputationGraph::computeGradients(const NodePtr& root)
{
	std::function<void(const NodePtr&, const mlCore::Tensor&)> backPropagate;

	// traverses the nodes tree and computes gradient in regard of every node
	backPropagate = [&backPropagate, this](const NodePtr& node, const mlCore::Tensor& cumulatedGradient)
	{
		if(this->_gradients.find(node) == this->_gradients.end())
		{
			this->_gradients.emplace(node, cumulatedGradient);
		}
		else
		{
			auto& grad = this->_gradients.at(node);
			grad = grad + cumulatedGradient;
		}

		if(const auto castedOp = std::dynamic_pointer_cast<Operator>(node))
		{
			const auto inputs = castedOp->getInputs();
			const auto derivatives = castedOp->computeDerivative(cumulatedGradient);

			if(inputs.size() != derivatives.size())
			{
				LOG_ERROR("ComputationGraph",
						  "Critical! Got different number of derivatives than inputs of an operator!");
				return;
			}

			for(std::size_t i = 0; i < inputs.size(); ++i)
			{
				backPropagate(inputs[i], derivatives[i]);
			}
		}
	};

	backPropagate(root, mlCore::Tensor(root->getValue().shape(), 1.0));
}
} // namespace autoDiff
