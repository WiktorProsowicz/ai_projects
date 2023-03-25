#ifndef MLCORE_GRAPHNODES_H
#define MLCORE_GRAPHNODES_H

#include <MLCore/BasicTensor.h>
#include <memory>

namespace mlCore
{
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
	static uint64_t nodesCount_;
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

	void setValue(const Tensor& tensor) override
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

/**
 * @brief Specifies how the Operator nodes are constructed, useful to compute derivative.
 * 
 */
enum class UnaryOperatorType : uint8_t
{

	/// activation functions
	RELU = 0,
	SIGMOID
};

class UnaryOperator : public Node
{
public:
	UnaryOperator(const NodePtr input, const UnaryOperatorType type)
		: Node(std::vector<size_t>{})
		, type_(type)
		, input_(input){};

	// tells the operator to compute its value based om its inputs
	void updateValue();

	const UnaryOperatorType getType() const
	{
		return type_;
	}

	NodePtr getInputs() const
	{
		return input_;
	}

private:
	UnaryOperatorType type_;
	NodePtr input_;
};

enum class BinaryOperatorType : uint8_t
{
	/// basic operations
	ADD = 0,
	SUBTRACT,
	MULTIPLY,
	DIVIDE,
	MATMUL,
	POWER,
};

class BinaryOperator : public Node
{
public:
	BinaryOperator(const NodePtr lhsInput, const NodePtr rhsInput, const BinaryOperatorType type)
		: Node(std::vector<size_t>{})
		, lhsInput_(lhsInput)
		, rhsInput_(rhsInput){};

	// tells the operator to compute its value based on its inputs
	void updateValue();

	const BinaryOperatorType getType() const
	{
		return type_;
	}

	std::pair<NodePtr, NodePtr> getInputs() const
	{
		return {lhsInput_, rhsInput_};
	}

private:
	BinaryOperatorType type_;
	const NodePtr lhsInput_;
	const NodePtr rhsInput_;
};

using NodePtr = std::shared_ptr<Node>;
using UnaryOperPtr = std::shared_ptr<UnaryOperator>;
using BinaryOperPtr = std::shared_ptr<BinaryOperator>;
using PlaceholderPtr = std::shared_ptr<Placeholder>;

} // namespace mlCore
#endif