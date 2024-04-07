#ifndef MLCORE_COMPUTATIONGRAPH_H
#define MLCORE_COMPUTATIONGRAPH_H

#include <map>
#include <string>
#include <vector>

#include "AutoDiff/GraphNodes.hpp"

namespace autoDiff
{

/**
 * @brief Class used to build tree of Nodes. Stores information about all of the parts
 * used in complex operation and can therefore accurately compute gradients.
 *
 */
class ComputationGraph
{

public:
	ComputationGraph() = default;

	ComputationGraph& operator=(const ComputationGraph&) = delete; // Copy assign
	ComputationGraph& operator=(ComputationGraph&&) = delete;	   // Move assign
	ComputationGraph(const ComputationGraph&) = delete;			   // Copy ctor
	ComputationGraph(ComputationGraph&&) = delete;				   // Move ctor

	~ComputationGraph() = default;

	/**
	 * @brief Gets status of the graph
	 *
	 * @return true Graph is declared to be able to extend
	 * @return false Graph should not be extended
	 */
	bool isActive() const noexcept
	{
		return _isActive;
	}

	/**
	 * @brief Erases all graph structure nodes
	 *
	 */
	void reset() noexcept
	{
		_nodes.clear();
		_gradients.clear();
	}

	/**
	 * @brief Cleans the graph from cumulated gradient.
	 *
	 */
	void clearGradients()
	{
		_gradients.clear();
	}

	/**
	 * @brief Enables adding nodes to the graph by friend Operations classes
	 *
	 */
	void activate() noexcept
	{
		_isActive = true;
	}

	/**
	 * @brief Blocks the graph and disallow it to extend
	 *
	 */
	void deactivate() noexcept
	{
		_isActive = false;
	}

	/**
	 * @brief Tells if a gradient computed with respect to the given node is available.
	 *
	 */
	bool hasGradient(const NodePtr& nodeName) const;

	/**
	 * @brief Returns gradient computed with respect to the given node.
	 *
	 */
	const mlCore::Tensor& getGradient(const NodePtr& node) const;

	/**
	 * @brief Goes through the graph starting from the primary leaves
	 *
	 * @param feedDict Stores values that should fill chosen placeholders. If not given, placeholders keep
	 * their old values.
	 */
	void forwardPass(const std::map<PlaceholderPtr, mlCore::Tensor>& feenodedDict = {});

	/**
	 * @brief Goes through the graph starting from the root and perform backward propagation
	 *
	 * @param root Starting node - the back-propagation will occur relatively to it
	 */
	void computeGradients(const NodePtr& root);

	/**
	 * @brief Adds new node to the graph.
	 *
	 * @param node Node to be added.
	 */
	void addNode(const NodePtr& node);

private:
	void _sortNodes();

private:
	bool _isActive = false;
	std::vector<NodePtr> _nodes;
	std::map<NodePtr, mlCore::Tensor> _gradients;
	bool _areNodesSorted = true;
};
} // namespace autoDiff

#endif
