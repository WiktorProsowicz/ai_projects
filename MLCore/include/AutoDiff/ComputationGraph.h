#ifndef MLCORE_COMPUTATIONGRAPH_H
#define MLCORE_COMPUTATIONGRAPH_H

#include <AutoDiff/GraphNodes.hpp>
#include <map>
#include <string>
#include <vector>

namespace mlCore
{
class UnaryOperations;
class BinaryOperations;
class NodesActivations;

/**
 * @brief Class used to build tree of Nodes. Stores information about all of the parts 
 * used in complex operation and can therefore accurately compute gradients. 
 * 
 */
class ComputationGraph
{
	friend class UnaryOperations;
	friend class BinaryOperations;
	friend class NodesActivations;

public:
	ComputationGraph() = default;

	ComputationGraph& operator=(const ComputationGraph&) = delete; // Copy assign
	ComputationGraph& operator=(ComputationGraph&&) = delete; // Move assign
	ComputationGraph(const ComputationGraph&) = delete; // Copy ctor
	ComputationGraph(ComputationGraph&&) = delete; // Move ctor

	/**
	 * @brief Gets status of the graph
	 * 
	 * @return true Graph is declared to be able to extend 
	 * @return false Graph should not be extended
	 */
	bool isActive() const
	{
		return isActive_;
	}

	/**
	 * @brief Erases all graph structure nodes
	 * 
	 */
	void reset() noexcept
	{
		nodes_.clear();
		gradients_.clear();
	}

	/**
	 * @brief Enables adding nodes to the graph by friend Operations classes
	 * 
	 */
	void activate()
	{
		isActive_ = false;
	}

	/**
	 * @brief Blocks the graph and disallow it to extend
	 * 
	 */
	void deactivate()
	{
		isActive_ = false;
	}

	/**
	 * @brief Tells if there is computed gradient with certain name
	 * 
	 * @param nodeName 
	 */
	bool hasGradient(const std::string& nodeName) const;

	/**
	 * @brief Gets gradient connected with the node with given name
	 * 
	 * @param nodeName
	 */
	const Tensor& getGradientByNodeName(const std::string& nodeName) const;

	/**
	 * @brief Tells if there is computed gradient with certain id
	 * 
	 * @param nodeId
	 */
	bool hasGradient(const size_t& nodeId) const;

	/**
	 * @brief Gets gradient connected with the node with given id
	 * 
	 * @param nodeId
	 */
	const Tensor& getGradientByNodeId(const size_t& nodeId) const;

	/**
	 * @brief Goes through the graph starting from the primary leaves
	 * 
	 * @param feedDict Stores values that should fill chosen placeholders. If not given, placeholders keep their old values.
	 */
	void forwardPass(const std::map<PlaceholderPtr, Tensor>& feedDict);

	/**
	 * @brief Goes through the graph starting from the root and perform backward propagation
	 * 
	 * @param root Starting node - the back-propagation will occur relatively to it
	 */
	void computeGradients(NodePtr root);

private:
	void addNode(NodePtr node);

	void sortNodes();

private:
	bool isActive_ = false;
	std::vector<NodePtr> nodes_ = {};
	std::map<NodePtr, Tensor> gradients_ = {};
	bool areNodesSorted_ = true;
};
} // namespace mlCore

#endif