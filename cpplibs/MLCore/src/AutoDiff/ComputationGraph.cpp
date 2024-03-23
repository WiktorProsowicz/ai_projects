#include <AutoDiff/ComputationGraph.h>

#include <set>

#include <AutoDiff/BinaryOperators/BinaryOperator.h>
#include <AutoDiff/UnaryOperators/UnaryOperator.h>

namespace mlCore::autoDiff
{

bool ComputationGraph::hasGradient(const size_t& nodeId) const
{
	return std::find_if(gradients_.begin(), gradients_.end(), [&nodeId](const std::pair<NodePtr, Tensor>& nodeGrad) {
			   return nodeGrad.first->getIndex() == nodeId;
		   }) == gradients_.end();
}

bool ComputationGraph::hasGradient(const std::string& nodeName) const
{
	return std::find_if(gradients_.begin(), gradients_.end(), [&nodeName](const std::pair<NodePtr, Tensor>& nodeGrad) {
			   return nodeGrad.first->getName() == nodeName;
		   }) == gradients_.end();
}

const Tensor& ComputationGraph::getGradientByNodeId(const size_t& nodeId) const
{
	return std::find_if(gradients_.begin(),
						gradients_.end(),
						[&nodeId](const std::pair<NodePtr, Tensor>& nodeGrad) { return nodeGrad.first->getIndex() == nodeId; })
		->second;
}

const Tensor& ComputationGraph::getGradientByNodeName(const std::string& nodeName) const
{
	return std::find_if(gradients_.begin(),
						gradients_.end(),
						[&nodeName](const std::pair<NodePtr, Tensor>& nodeGrad) { return nodeGrad.first->getName() == nodeName; })
		->second;
}

void ComputationGraph::addNode(const NodePtr node)
{
	if(!isActive_)
	{
		LOG_WARN("ComputationGraph", "Cannot add node to the graph which is not active.");
		return;
	}

	areNodesSorted_ = false;
	nodes_.push_back(node);
}

void ComputationGraph::_sortNodes()
{

	std::set<NodePtr> nodesWithParent;

	for(const auto& node : nodes_)
	{
		if(const auto castedBinaryOp = std::dynamic_pointer_cast<binaryOperators::BinaryOperator>(node))
		{
			const auto& [lhs, rhs] = castedBinaryOp->getInputs();

			nodesWithParent.insert(lhs);
			nodesWithParent.insert(rhs);
		}
		else if(const auto castedUnaryOp = std::dynamic_pointer_cast<unaryOperators::UnaryOperator>(node))
		{
			nodesWithParent.insert(castedUnaryOp->getInput());
		}
	}

	std::set<NodePtr> orphanedNodes;

	std::set_difference(nodes_.cbegin(),
						nodes_.cend(),
						nodesWithParent.cbegin(),
						nodesWithParent.cend(),
						std::inserter(orphanedNodes, orphanedNodes.end()));

	std::vector<NodePtr> newNodes;

	// recursively goes down the tree in a DFS manner and adds nodes to newNodes so that all inputs can be assigned before the operators
	std::function<void(const NodePtr)> traverseTree;
	traverseTree = [&traverseTree, &newNodes](const NodePtr node) {
		if(const auto castedBinaryOp = std::dynamic_pointer_cast<binaryOperators::BinaryOperator>(node))
		{
			const auto& [lhs, rhs] = castedBinaryOp->getInputs();
			traverseTree(lhs);
			traverseTree(rhs);
		}
		else if(const auto castedUnaryOp = std::dynamic_pointer_cast<unaryOperators::UnaryOperator>(node))
		{
			traverseTree(castedUnaryOp->getInput());
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

	nodes_ = newNodes;

	areNodesSorted_ = true;
}

void ComputationGraph::forwardPass(const std::map<PlaceholderPtr, Tensor>& feedDict)
{
	if(!areNodesSorted_)
	{
		_sortNodes();
	}

	for(const auto& node : nodes_)
	{
		if(const auto placeholder = std::dynamic_pointer_cast<Placeholder>(node);
		   placeholder && (feedDict.find(placeholder) != feedDict.end()))
		{
			placeholder->getValue() = feedDict.at(placeholder);
		}
		else if(const auto binaryOper = std::dynamic_pointer_cast<binaryOperators::BinaryOperator>(node))
		{
			binaryOper->updateValue();
		}
		else if(const auto unaryOper = std::dynamic_pointer_cast<unaryOperators::UnaryOperator>(node))
		{
			unaryOper->updateValue();
		}
	}
}

void ComputationGraph::computeGradients(const NodePtr root)
{
	if(!areNodesSorted_)
	{
		_sortNodes();
	}

	std::function<void(const NodePtr, const Tensor&)> backPropagate;

	// traverses the nodes tree and computes gradient in regard of every node
	backPropagate = [&backPropagate, this](const NodePtr node, const Tensor& cumulatedGradient) {
		if(this->gradients_.find(node) == this->gradients_.end())
		{
			this->gradients_.emplace(node, cumulatedGradient);
		}
		else
		{
			auto& grad = this->gradients_.at(node);
			grad = grad + cumulatedGradient;
		}

		if(const auto castedUnary = std::dynamic_pointer_cast<unaryOperators::UnaryOperator>(node))
		{
			const auto input = castedUnary->getInput();
			const auto derivative = castedUnary->computeDerivative(cumulatedGradient);
			backPropagate(input, cumulatedGradient);
		}
		else if(const auto castedBinary = std::dynamic_pointer_cast<binaryOperators::BinaryOperator>(node))
		{
			const auto [lInput, rInput] = castedBinary->getInputs();
			const auto [lDerivative, rDerivative] = castedBinary->computeDerivative(cumulatedGradient);

			backPropagate(lInput, lDerivative);
			backPropagate(rInput, rDerivative);
		}
	};

	backPropagate(root, Tensor(root->getValue().shape(), 1.0));
}
} // namespace mlCore::autoDiff
