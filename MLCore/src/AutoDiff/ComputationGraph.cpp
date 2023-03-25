#include <AutoDiff/ComputationGraph.h>

namespace mlCore
{
ComputationGraph::ComputationGraph()
	: isActive_(false)
	, areNodesSorted_(true)
{ }

bool ComputationGraph::hasGradient(const size_t& nodeId) const
{
	return std::find_if(gradients_.begin(),
						gradients_.end(),
						[&nodeId](const std::pair<NodePtr, Tensor>& nodeGrad) {
							return nodeGrad.first->getIndex() == nodeId;
						}) == gradients_.end();
}

bool ComputationGraph::hasGradient(const std::string& nodeName) const
{
	return std::find_if(gradients_.begin(),
						gradients_.end(),
						[&nodeName](const std::pair<NodePtr, Tensor>& nodeGrad) {
							return nodeGrad.first->getName() == nodeName;
						}) == gradients_.end();
}

const Tensor& ComputationGraph::getGradientByNodeId(const size_t& nodeId) const
{
	return std::find_if(gradients_.begin(),
						gradients_.end(),
						[&nodeId](const std::pair<NodePtr, Tensor>& nodeGrad) {
							return nodeGrad.first->getIndex() == nodeId;
						})
		->second;
}

const Tensor& ComputationGraph::getGradientByNodeName(const std::string& nodeName) const
{
	return std::find_if(gradients_.begin(),
						gradients_.end(),
						[&nodeName](const std::pair<NodePtr, Tensor>& nodeGrad) {
							return nodeGrad.first->getName() == nodeName;
						})
		->second;
}

void ComputationGraph::addNode(const NodePtr node)
{
	areNodesSorted_ = false;
	nodes_.push_back(node);
}

void ComputationGraph::sortNodes()
{
	std::vector<NodePtr> newNodes;

	// recursively goes down the tree in a DFS manner and adds nodes to newNodes so that all inputs can be assigned before the operators
	std::function<void(const NodePtr)> traverseTree;
	traverseTree = [&traverseTree, &newNodes](const NodePtr node) {
		if(const auto castedBinaryOp = std::dynamic_pointer_cast<BinaryOperator>(node))
		{
			const auto& [lhs, rhs] = castedBinaryOp->getInputs();
			traverseTree(lhs);
			traverseTree(rhs);
		}
		else if(const auto castedUnaryOp = std::dynamic_pointer_cast<UnaryOperator>(node))
		{
			traverseTree(castedUnaryOp->getInputs());
		}

		if(std::find(newNodes.begin(), newNodes.end(), node) == newNodes.end())
			newNodes.push_back(node);
	};

	for(auto nodesIter = nodes_.crbegin(); nodesIter < nodes_.crend(); nodesIter++)
	{
		traverseTree(*nodesIter);
	}

	areNodesSorted_ = true;
}

void ComputationGraph::forwardPass(const std::map<PlaceholderPtr, Tensor>& feedDict)
{
	if(!areNodesSorted_)
		sortNodes();

	for(const auto& node : nodes_)
	{
		if(const auto placeholder = std::dynamic_pointer_cast<Placeholder>(node);
		   placeholder && (feedDict.find(placeholder) != feedDict.end()))
		{
			placeholder->setValue(feedDict.at(placeholder));
		}
		else if(const auto binaryOper = std::dynamic_pointer_cast<BinaryOperator>(node))
		{
			binaryOper->updateValue();
		}
		else if(const auto unaryOper = std::dynamic_pointer_cast<UnaryOperator>(node))
		{
			unaryOper->updateValue();
		}
	}
}

void ComputationGraph::computeGradients(const NodePtr root)
{
	if(!areNodesSorted_)
		sortNodes();

	std::function<void(const NodePtr, const Tensor&)> backPropagate;
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

		if(const auto castedUnary = std::dynamic_pointer_cast<UnaryOperator>(node))
		{
			const auto input = castedUnary->getInputs();
			const auto derivative = DerivativeExtractor{}(castedUnary);
			backPropagate(input, cumulatedGradient * derivative);
		}
		else if(const auto castedBinary = std::dynamic_pointer_cast<BinaryOperator>(node))
		{
			const auto [lInput, rInput] = castedBinary->getInputs();
			const auto [lDerivative, rDerivative] = DerivativeExtractor{}(castedBinary);
			backPropagate(lInput, cumulatedGradient * lDerivative);
			backPropagate(rInput, cumulatedGradient * rDerivative);
		}
	};

	backPropagate(root, Tensor(std::vector<size_t>{}, 1));
}
} // namespace mlCore