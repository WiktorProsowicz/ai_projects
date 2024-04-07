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

#include "AutoDiff/GraphNodes.hpp"
#include "MLCore/BasicTensor.h"

namespace autoDiff
{

bool ComputationGraph::hasGradient(const std::string& nodeName) const
{
	return std::find_if(_gradients.begin(),
						_gradients.end(),
						[&nodeName](const std::pair<NodePtr, mlCore::Tensor>& nodeGrad)
						{ return nodeGrad.first->getName() == nodeName; }) == _gradients.end();
}

const mlCore::Tensor& ComputationGraph::getGradientByNodeName(const std::string& nodeName) const
{
	return std::find_if(_gradients.begin(),
						_gradients.end(),
						[&nodeName](const std::pair<NodePtr, mlCore::Tensor>& nodeGrad)
						{ return nodeGrad.first->getName() == nodeName; })
		->second;
}

void ComputationGraph::addNode(const NodePtr& node)
{
	if(!_isActive)
	{
		LOG_WARN("ComputationGraph", "Cannot add node to the graph which is not active.");
		return;
	}

	_areNodesSorted = false;
	_nodes.push_back(node);
}

void ComputationGraph::_sortNodes()
{

	std::set<NodePtr> nodesWithParent;

	for(const auto& node : _nodes)
	{
		if(const auto castedOperator = std::dynamic_pointer_cast<Operator>(node))
		{
			nodesWithParent.insert(castedOperator->getInputs().begin(), castedOperator->getInputs().end());
		}
	}

	std::set<NodePtr> orphanedNodes;

	std::set_difference(_nodes.cbegin(),
						_nodes.cend(),
						nodesWithParent.cbegin(),
						nodesWithParent.cend(),
						std::inserter(orphanedNodes, orphanedNodes.end()));

	std::vector<NodePtr> newNodes;

	// recursively goes down the tree in a DFS manner and adds nodes to newNodes so that all inputs can be
	// assigned before the operators
	std::function<void(const NodePtr&)> traverseTree;
	traverseTree = [&traverseTree, &newNodes](const NodePtr& node)
	{
		if(const auto castedOp = std::dynamic_pointer_cast<Operator>(node))
		{
			std::for_each(castedOp->getInputs().begin(), castedOp->getInputs().end(), traverseTree);
		}

		if(std::find(newNodes.begin(), newNodes.end(), node) == newNodes.end())
		{
			newNodes.push_back(node);
		}
	};

	for(const auto& orphanedNode : orphanedNodes)
	{
		traverseTree(orphanedNode);
	}

	_nodes = newNodes;

	_areNodesSorted = true;
}

void ComputationGraph::forwardPass(const std::map<PlaceholderPtr, mlCore::Tensor>& feedDict)
{
	if(!_areNodesSorted)
	{
		_sortNodes();
	}

	for(const auto& node : _nodes)
	{
		if(const auto placeholder = std::dynamic_pointer_cast<Placeholder>(node);
		   placeholder && (feedDict.find(placeholder) != feedDict.end()))
		{
			placeholder->putValue(feedDict.at(placeholder));
		}
		else if(const auto castedOp = std::dynamic_pointer_cast<Operator>(node))
		{
			castedOp->updateValue();
		}
	}
}

void ComputationGraph::computeGradients(const NodePtr& root)
{
	if(!_areNodesSorted)
	{
		_sortNodes();
	}

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
