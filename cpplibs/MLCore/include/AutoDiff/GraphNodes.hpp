#ifndef MLCORE_GRAPHNODES_H
#define MLCORE_GRAPHNODES_H

#include <memory>

#include "MLCore/BasicTensor.h"

/**
 * @brief Classes representing nodes in ComputationGraphs. Nodes hold mlCore::Tensors and can be linked to
 * each other.
 *
 */
namespace autoDiff
{

class Node;
class Placeholder;
class Variable;
class Constant;
class Operator;

using NodePtr = std::shared_ptr<Node>;
using PlaceholderPtr = std::shared_ptr<Placeholder>;
using VariablePtr = std::shared_ptr<Variable>;
using ConstantPtr = std::shared_ptr<Constant>;
using OperatorPtr = std::shared_ptr<Operator>;

/**
 * @brief The base class of all computation graph nodes.
 *
 * @details The graph nodes are building blocks of computation graph and represent particular parts of
 * possibly complex calculation process. The nodes are intended to be linked to each other and be used as
 * layers' weights, return values of operators etc.
 *
 */
class Node
{
public:
	Node() = default;

	Node(const Node&) = delete;
	Node(Node&&) = delete;
	Node& operator=(const Node&) = delete;
	Node& operator=(Node&&) = delete;

	virtual ~Node() = default;

	/**
	 * @brief Returns node's name identifier.
	 *
	 */
	const std::string& getName() const
	{
		return _name;
	}

	/**
	 * @brief Sets node's name identifier.
	 *
	 */
	void setName(const std::string& name)
	{
		_name = name;
	}

	/**
	 * @brief Returns the internal mlCore::Tensor wrapped by the node.
	 *
	 * The shape and data of the wrapped mlCore::Tensor depends on the specific node's behavior.
	 *
	 */
	virtual const mlCore::Tensor& getValue() const = 0;

	/**
	 * @brief Spawns a copy of the node.
	 *
	 * The copy shall has the same shape and data as the original node.
	 *
	 * @return NodePtr A shared pointer to the copy of the node.
	 */
	virtual NodePtr copy() const = 0;

	/**
	 * @brief Returns the shape of the value being on the output port of the node.
	 *
	 * It may be simply the shape of the internal value or shape of the value computed via a compilcated
	 * internal operation depending on the concrete type of node.
	 *
	 */
	virtual const std::vector<size_t>& getOutputShape() const = 0;

private:
	std::string _name{};
};

/**
 * @brief Represents a node with mutable value.
 *
 * @details The value can be assigned, changed, therefore variables can be used as weights of neural networks.
 * While performing back-propagation, the derivative regarding any variable should be computed via chain-rule,
 * since this type of node is not divisible.
 *
 */
class Variable final : public Node
{
public:
	Variable() = delete;

	/**
	 * @brief Creates the variable, giving it an initial value.
	 *
	 * @param initValue Initial value of the variable.
	 */
	explicit Variable(mlCore::Tensor initValue)
		: _value(std::move(initValue)){};

	Variable(const Variable&) = delete;
	Variable(Variable&&) = delete;
	Variable& operator=(const Variable&) = delete;
	Variable& operator=(Variable&&) = delete;

	~Variable() override = default;

	const mlCore::Tensor& getValue() const override
	{
		return _value;
	}

	/**
	 * @brief Returns a 'trivial' copy of the variable.
	 *
	 */
	NodePtr copy() const override
	{
		return std::make_shared<Variable>(_value);
	}

	const std::vector<size_t>& getOutputShape() const override
	{
		return _value.shape();
	}

	/**
	 * @brief Sets the value of the variable.
	 *
	 * The new value is passed value so that usually one copy is made. It is useful in situations where the
	 * variable is updated with its own previous value.
	 *
	 * @param value New value for the variable.
	 *
	 * @example
	 *
	 * Variable v(mlCore::Tensor({1, 2, 3}, 0));
	 *
	 * v.setValue(v.getValue() + 1);
	 */
	void setValue(mlCore::Tensor value)
	{
		_value = std::move(value);
	}

private:
	mlCore::Tensor _value;
};

/**
 * @brief Represents a node with constant value, that cannot be changed in any way.
 *
 * @details While computing derivative with regard to a constant, the result is shall be zeroed.
 *
 */
class Constant final : public Node
{
public:
	Constant() = delete;

	/**
	 * @brief Creates the constant, giving it an initial value.
	 *
	 * @param initValue Initial value of the constant.
	 */
	explicit Constant(mlCore::Tensor initValue)
		: _value(std::move(initValue)){};

	Constant(const Constant&) = delete;
	Constant(Constant&&) = delete;
	Constant& operator=(const Constant&) = delete;
	Constant& operator=(Constant&&) = delete;

	~Constant() override = default;

	const mlCore::Tensor& getValue() const override
	{
		return _value;
	}

	/**
	 * @brief Returns a 'trivial' copy of the constant.
	 *
	 */
	NodePtr copy() const override
	{
		return std::make_shared<Constant>(_value);
	}

	const std::vector<size_t>& getOutputShape() const override
	{
		return _value.shape();
	}

private:
	mlCore::Tensor _value;
};

/**
 * @brief Provides semantics for graph element holding external data.
 *
 */
class Placeholder final : public Node
{
public:
	Placeholder() = delete;

	/**
	 * @brief Creates the placeholder, giving it an initial value.
	 *
	 * @param initValue Initial value of the placeholder.
	 */
	explicit Placeholder(const mlCore::Tensor& value)
		: _value(value)
	{}

	Placeholder(const Placeholder&) = delete;
	Placeholder(Placeholder&&) = delete;
	Placeholder& operator=(const Placeholder&) = delete;
	Placeholder& operator=(Placeholder&&) = delete;

	~Placeholder() override = default;

	const mlCore::Tensor& getValue() const override
	{
		return _value.get();
	}

	/**
	 * @brief Returns a 'trivial' copy of the placeholder.
	 *
	 */
	NodePtr copy() const override
	{
		auto copiedPlaceholder = std::make_shared<Placeholder>(_value.get());

		return copiedPlaceholder;
	}

	/**
	 * @brief Links a given mlCore::Tensor to the internal reference of the placeholder.
	 *
	 * @param value mlCore::Tensor to be put in the placeholder.
	 */
	void putValue(const mlCore::Tensor& value)
	{
		_value = value;
	}

	const std::vector<size_t>& getOutputShape() const override
	{
		return _value.get().shape();
	}

private:
	std::reference_wrapper<const mlCore::Tensor> _value;
};

/**
 * @brief Represents a node that performs some operation on its input nodes.
 *
 * @details The operator is a node that takes a list of input nodes and produces a single output node.
 * The operator is intended to be used as a core of a layer in neural networks, or as a part of more complex
 * computation graph.
 *
 */
class Operator : public Node
{
public:
	Operator() = delete;

	/**
	 * @brief Creates the operator, giving it a list of input nodes.
	 *
	 * The internal value of the operator is initialized with the computed shape depending on the type of
	 * operator and input nodes. It should be then updated calling the `updateValue` method.
	 *
	 * @param inputs List of nodes on which the operator depends.
	 */
	explicit Operator(std::vector<NodePtr> inputs)
		: _inputs(std::move(inputs))
	{}

	Operator(const Operator&) = delete;
	Operator(Operator&&) = delete;
	Operator& operator=(const Operator&) = delete;
	Operator& operator=(Operator&&) = delete;

	~Operator() override = default;

	virtual const mlCore::Tensor& getValue() const = 0;

	/**
	 * @brief Creates a copy of the operator.
	 *
	 * The type of the copy depends on the concrete operator class. The copy should have the same inputs as
	 * the base op and also the same internal state.
	 *
	 */
	virtual NodePtr copy() const = 0;

	/**
	 * @brief Updates the internal value of the operator.
	 *
	 * The method should be called after the input nodes have been updated.
	 *
	 */
	virtual void updateValue() = 0;

	/**
	 * @brief Computes derivative of the operator with respect to its inputs. Then applies the
	 * `outerDerivative` for nested expressions differentiation. The computing of the result assumes that the
	 * internal value of the operator is updated.
	 *
	 * @param outerDerivative The derivative of outer expression with respect to the operator. Used to
	 * differentiate nested expressions. Applying the `outerDerivative` to the result depends of the concrete
	 * operator class.
	 *
	 * @return Derivatives of the operator with respect to the inputs.
	 */
	virtual std::vector<mlCore::Tensor> computeDerivative(const mlCore::Tensor& outerDerivative) const = 0;

	/**
	 * @brief Computes derivative of the operator with respect to its inputs locally regardless of the
	 * context. The computing of the result assumes that the internal value of the operator is updated.
	 *
	 * @return Derivative of the operator with respect to the inputs.
	 */
	virtual std::vector<mlCore::Tensor> computeDirectDerivative() const = 0;

	/**
	 * @brief Returns the list of input nodes.
	 *
	 */
	const std::vector<NodePtr>& getInputs() const
	{
		return _inputs;
	}

private:
	std::vector<NodePtr> _inputs;
};

} // namespace autoDiff
#endif
