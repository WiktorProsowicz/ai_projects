#ifndef UNARYOPERATORS_IUNARYOPERATOR_H
#define UNARYOPERATORS_IUNARYOPERATOR_H

#include "AutoDiff/GraphNodes.hpp"

namespace mlCore::autoDiff::unaryOperators
{
/**
 * @brief Represents a result of modification of a single node. Its internal processing depends on its
 * subclass, wraps TensorFunctions algorithms.
 *
 */
class UnaryOperator : public Node
{
public:
	UnaryOperator(const NodePtr input)
		: Node(std::vector<size_t>{})
		, input_(input){};

	/**
	 * @brief Tells the operator to compute its value based om its input.
	 *
	 */
	virtual void updateValue() = 0;

	/**
	 * @brief Computes derivative of the operator with respect to its input. Applies `outerDerivative` to the
	 * result for chain rule purposes. The computing of the result assumes that the internal value of the
	 * operator is updated.
	 *
	 * @param outerDerivative Derivative of external expression with respect to the operator. Applying the
	 * `outerDerivative` to the result depends of the concrete operator class.
	 * @return Derivative of the operator with respect to the input.
	 */
	virtual Tensor computeDerivative(const Tensor& outerDerivative) const = 0;

	/**
	 * @brief Computes derivative of the operator with respect to its input locally with no regards to the
	 * context. The computing of the result assumes that the internal value of the operator is updated.
	 *
	 * @return Derivative of the operator with respect to the input.
	 */
	virtual Tensor computeDirectDerivative() const = 0;

	NodePtr getInput() const
	{
		return input_;
	}

protected:
	NodePtr input_;
};

using UnaryOperatorPtr = std::shared_ptr<UnaryOperator>;

} // namespace mlCore::autoDiff::unaryOperators

#endif
