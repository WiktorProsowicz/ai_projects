#ifndef MLCORE_IBINARYOPERATOR_H
#define MLCORE_IBINARYOPERATOR_H

#include <optional>

#include <AutoDiff/GraphNodes.hpp>

namespace mlCore::autoDiff::binaryOperators
{
/**
 * @brief Represents a result of operation on two nodes. Its internal processing depends on its type, wraps
 * TensorFunctions algorithms.
 *
 */
class BinaryOperator : public Node
{
public:
	BinaryOperator(const NodePtr lhsInput, const NodePtr rhsInput)
		: Node(std::vector<size_t>{})
		, lhsInput_(lhsInput)
		, rhsInput_(rhsInput){};

	/**
	 * @brief Tells the operator to compute its value based on its inputs.
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
	 * @return Derivatives of the operator with respect to left an right input.
	 */
	virtual std::pair<Tensor, Tensor> computeDerivative(const Tensor& outerDerivative) const = 0;

	/**
	 * @brief Computes derivative of the operator with respect to its inputs locally with no regards to the
	 * context. The computing of the result assumes that the internal value of the operator is updated.
	 *
	 * @return Derivative of the operator with respect to left and right input.
	 */
	virtual std::pair<Tensor, Tensor> computeDirectDerivative() const = 0;

	std::pair<NodePtr, NodePtr> getInputs() const
	{
		return {lhsInput_, rhsInput_};
	}

protected:
	const NodePtr lhsInput_;
	const NodePtr rhsInput_;
};

using BinaryOperatorPtr = std::shared_ptr<BinaryOperator>;

} // namespace mlCore::autoDiff::binaryOperators

#endif
