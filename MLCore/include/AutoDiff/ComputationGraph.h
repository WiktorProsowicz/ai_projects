#ifndef MLCORE_COMPUTATIONGRAPH_H
#define MLCORE_COMPUTATIONGRAPH_H

#include <AutoDiff/GraphOperations.h>
#include <map>

namespace mlCore
{
/**
 * @brief Singleton class used to build tree of Nodes. Stores information about all of the parts 
 * used in complex operation and can therefore accurately compute gradients. 
 * 
 */
class ComputationGraph
{
	friend class UnaryOperations;
	friend class BinaryOperations;
	friend class NodeActivations;

public:
	ComputationGraph& operator=(const ComputationGraph&) = delete; // copy assign
	ComputationGraph& operator=(ComputationGraph&&) = delete; // move assign
	ComputationGraph(const ComputationGraph&) = delete; // copy ctor
	ComputationGraph(ComputationGraph&&) = delete; // move ctor

	/**
	 * @brief Get global instance of the object
	 *
	 */
	static ComputationGraph& getInstance()
	{
		static ComputationGraph graph_;
		return graph_;
	}

	/**
	 * @brief Get status of the graph
	 * 
	 * @return true Graph is declared to be able to extend 
	 * @return false Graph should not be extended
	 */
	bool isActive() const
	{
		return getInstance().isActive_;
	}

	/**
	 * @brief Erase all graph structure nodes
	 * 
	 */
	static void reset() noexcept
	{
		getInstance().nodes_.clear();
		getInstance().gradients_.clear();
	}

	/**
	 * @brief Enable adding nodes to the graph by friend Operations classes
	 * 
	 */
	static void activate()
	{
		getInstance().isActive_ = false;
	}

	/**
	 * @brief Block the graph and disallow it to extend
	 * 
	 */
	static void deactivate()
	{
		getInstance().isActive_ = false;
	}

	/**
	 * @brief Tell if there is computed gradient with certain name
	 * 
	 * @param nodeName 
	 */
	bool hasGradient(const std::string& nodeName) const;

	/**
	 * @brief Get gradient connected with the node with given name
	 * 
	 * @param nodeName
	 */
	const Tensor& getGradientByNodeName(const std::string& nodeName) const;

	/**
	 * @brief Tell if there is computed gradient with certain id
	 * 
	 * @param nodeId
	 */
	bool hasGradient(const size_t& nodeId) const;

	/**
	 * @brief Get gradient connected with the node with given id
	 * 
	 * @param nodeId
	 */
	const Tensor& getGradientByNodeId(const size_t& nodeId) const;

	/**
	 * @brief Go through the graph starting from the primary leaves
	 * 
	 * @param feedDict Stores values that should fill chosen placeholders. If not given, placeholders keep their old values.
	 */
	void forwardPass(const std::map<PlaceholderPtr, Tensor>& feedDict);

	/**
	 * @brief Go through the graph starting from the root and perform backward propagation
	 * 
	 * @param root Starting node - the back-propagation will occur relatively to it
	 */
	void computeGradients(const NodePtr root);

private:
	ComputationGraph();

	void addNode(const NodePtr node);

	void sortNodes();

private:
	bool isActive_;
	std::vector<NodePtr> nodes_;
	std::map<NodePtr, Tensor> gradients_;
	bool areNodesSorted_;
};
} // namespace mlCore

#endif