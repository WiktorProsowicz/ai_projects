#ifndef MLCORE_GRAPHNODES_H
#define MLCORE_GRAPHNODES_H

#include <MLCore/BasicTensor.h>
#include <memory>

/**
 * @brief Classes representing nodes in ComputationGraphs. Nodes hold tensors and can be linked to each other.
 * 
 */
namespace mlCore::autoDiff
{

class Node;
class Placeholder;
class Variable;
class Constant;

using NodePtr = std::shared_ptr<Node>;
using PlaceholderPtr = std::shared_ptr<Placeholder>;
using VariablePtr = std::shared_ptr<Variable>;
using ConstantPtr = std::shared_ptr<Constant>;

/**
 * @brief Mother class of computation graph nodes.
 * 
 */
class Node
{
public:
	Node() = delete;
	Node(const Tensor& tensor)
		: index_(nodesCount_++)
		, value_(tensor){};

	virtual ~Node() = default;

	Tensor& getValue()
	{
		return value_;
	}

	const uint64_t& getIndex() const
	{
		return index_;
	}

	const std::string& getName() const
	{
		return name_;
	}

	void setName(const std::string& name)
	{
		name_ = name;
	}

protected:
	uint64_t index_;
	static inline uint64_t nodesCount_ = 0;
	Tensor value_;
	std::string name_ = "";
};

/**
 * @brief A class to be inserted as i.e. weight matrix, bias etc
 * 
 */
class Variable : public Node
{
public:
	Variable()
		: Node(std::vector<size_t>{}){};
	Variable(const Tensor& tensor)
		: Node(tensor){};
};

/**
 * @brief Its value cannot be assigned, changed. Computed derivative is always zero.
 * 
 */
class Constant : public Node
{
public:
	Constant() = delete;
	Constant(const Tensor& tensor)
		: Node(tensor){};
};

/**
 * @brief Created to provide proper semantics for graph element holding external data.
 * 
 */
class Placeholder : public Node
{
public:
	Placeholder(const std::vector<size_t>& shape = {})
		: Node(shape){};
};

} // namespace mlCore::autoDiff
#endif