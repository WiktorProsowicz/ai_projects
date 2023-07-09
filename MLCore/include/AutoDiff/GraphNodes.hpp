#ifndef MLCORE_GRAPHNODES_H
#define MLCORE_GRAPHNODES_H

#include <MLCore/BasicTensor.h>
#include <memory>

/**
 * @brief Classes representing nodes in ComputationGraphs. Nodes hold tensors and can be linked to each other.
 * 
 */
namespace mlCore
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
	Node(const Tensor& tensor, const std::string& name = "")
		: index_(nodesCount_++)
		, value_(tensor)
		, name_(name){};

	virtual ~Node() = default;

	const Tensor& getValue() const
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

	virtual void setValue(const Tensor& tensor)
	{
		value_ = tensor;
	}

protected:
	uint64_t index_;
	static inline uint64_t nodesCount_ = 0;
	Tensor value_;
	const std::string name_;
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
	Variable(const Tensor& tensor, const std::string& name = "")
		: Node(tensor, name){};
};

/**
 * @brief Its value cannot be assigned, changed. Computed derivative is always zero.
 * 
 */
class Constant : public Node
{
public:
	Constant() = delete;
	Constant(const Tensor& tensor, const std::string& name = "")
		: Node(tensor, name){};

	void setValue(const Tensor& /*tensor*/) override
	{
		LOG_WARN("GraphNodes", "Attempt to assign value to constant");
	}
};

/**
 * @brief Created to provide proper semantics for graph element holding external data.
 * 
 */
class Placeholder : public Node
{
public:
	Placeholder()
		: Node(std::vector<size_t>{}){};
};

} // namespace mlCore
#endif