#ifndef MLCORE_INCLUDE_AUTODIFF_COMPUTATIONGRAPH_H
#define MLCORE_INCLUDE_AUTODIFF_COMPUTATIONGRAPH_H

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "AutoDiff/GraphNodes.hpp"

namespace autoDiff
{
/**
 * @brief Contains computation graph parameters.
 *
 */
struct ComputationGraphConfig
{
	bool useMultithreading;
};

/**
 * @brief Represents a directed graph containing nodes consisting on a structure of operations.
 *
 * @details The main role of the graph is to keep track of computation of the values according to linked
 * nodes. The information about the graph structure allows to compute gradients of an operator with respect to
 * its both direct and indirect inputs. Gradients computed in course of back-propagation are stored in the
 * graph and can be retrieved in order to perform optimization.
 *
 * ComputationGraph is intended to be used according to the following pattern:
 * 1. Perform a complex set of computations on GraphNodes structures.
 * 2. Set the obtained results as the root nodes of the graph. The results can be perceived as the output of a
 * complex operation, one may pick an arbitrary node from the structure of computations.
 * 3. Perform forward pass on the graph. The graph will compute the values of all nodes in the graph, starting
 * from root nodes.
 * 4. Perform backward pass on the graph. The graph will compute the gradients with respect to nodes specified
 * by the caller. These nodes could be trainable weights of the model.
 * 5. Retrieve the gradients from the graph and use them to perform optimization.
 *
 */
class ComputationGraph
{
public:
	/**
	 * @brief Constructs a computation graph with given configuration.
	 *
	 */
	ComputationGraph(const ComputationGraphConfig& config);

	ComputationGraph() = delete;

	ComputationGraph& operator=(const ComputationGraph&) = delete; // Copy assign
	ComputationGraph& operator=(ComputationGraph&&) = delete;	   // Move assign
	ComputationGraph(const ComputationGraph&) = delete;			   // Copy ctor
	ComputationGraph(ComputationGraph&&) = delete;				   // Move ctor

	~ComputationGraph() = default;

	/**
	 * @brief Cleans the graph from cumulated gradients.
	 *
	 * This operation should be performed after each optimization step. Since the gradients computed with
	 * respect to a particular node are cumulated, consecutive calls to the graph's backward-pass algorithm
	 * update the gradients instead of replacing them.
	 *
	 */
	void clearGradients()
	{
		_gradients.clear();
	}

	/**
	 * @brief Tells if a gradient computed with respect to the given node is available.
	 *
	 */
	bool hasGradient(const NodePtr& nodeName) const;

	/**
	 * @brief Returns stored gradient computed with respect to a given node.
	 *
	 */
	const mlCore::Tensor& getGradient(const NodePtr& node) const;

	/**
	 * @brief Performs operations between nodes spanned by the graph.
	 *
	 * The computation is performed starting with depth-first search of the leaf nodes to ensure the correct
	 * order of graph traversal.
	 *
	 */
	void forwardPass();

	/**
	 * @brief Traverses the graph starting from the root and performs backward propagation.
	 *
	 * The gradients are cumulated each time the derivative is computed with respect to a given node. The
	 * `root` node is allowed to be different then the root of the graph. Is could be for example the result
	 * of a loss function.
	 *
	 * @param root Starting node - the back-propagation will occur relatively to it
	 */
	void computeGradients(const NodePtr& root);

	/**
	 * @brief Sets a given node as the root of the graph.
	 *
	 * The root determines what part of an existing graph is spanned by the class.
	 *
	 */
	void setRoot(const NodePtr& root);

	/**
	 * @brief Sets the nodes for which the graph shall store gradients.
	 *
	 * The nodes could be for example trainable weights of a model.
	 */
	void setDifferentiableNodes(const std::vector<NodePtr>& nodes);

	/**
	 * @brief Creates the visualization of the graph in the DOT format.
	 *
	 * The visualization contains nodes representing the operations or values and edges representing the data
	 * flow. Each node is labelled with the basic information about it, such as the output shape or name.
	 *
	 */
	std::string serializeToDot() const;

private:
	ComputationGraphConfig _config;
	NodePtr _root{};
	std::vector<NodePtr> _differentiableNodes{};
	std::map<NodePtr, mlCore::Tensor> _gradients{};
};
} // namespace autoDiff

#endif
