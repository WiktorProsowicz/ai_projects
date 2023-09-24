#ifndef MLCORE_INCLUDE_AUTODIFF_MULTIPLEOPERATORS_MULTIPLEOPERATOR_H
#define MLCORE_INCLUDE_AUTODIFF_MULTIPLEOPERATORS_MULTIPLEOPERATOR_H

#include <AutoDiff/GraphNodes.hpp>

namespace mlCore::autoDiff::multipleOperators
{
/**
 * @brief Represents a result of operation of multiple input nodes. 
 * 
 * Internal processing depends on the concrete implementation of the operator. 
 * It is also the concrete class' task to validate number of the given inputs and their shapes.
 * 
 */
class MultipleOperator : public Node
{
public:
	/**
     * @brief Created new operator and sets its inputs.
     * 
     * @param inputs Initial inputs of the operator.
     */
	MultipleOperator(const std::vector<NodePtr>& inputs)
		: Node(std::vector<size_t>{})
		, inputs_(inputs)
	{ }

	~MultipleOperator() override = default; /// Default destructor.

	/**
     * @brief Assigns the value of the operator based on its inputs and represented algorithm.
     * 
     */
	virtual void updateValue() = 0;

	/**
	 * @brief Computes derivative of the operator with respect to its inputs. Then applies the `outerDerivative` for nested expressions differentiation.
	 * The computing of the result assumes that the internal value of the operator is updated.
	 * 
	 * @param outerDerivative The derivative of outer expression with respect to the operator. Used to differentiate nested expressions.
	 * Applying the `outerDerivative` to the result depends of the concrete operator class.
	 * @return Derivatives of the operator with respect to left an right input.
	 */
	virtual std::vector<Tensor> computeDerivative(const Tensor& outerDerivative) const = 0;

	/**
	 * @brief Computes derivative of the operator with respect to its inputs locally with no regards to the context.
	 * The computing of the result assumes that the internal value of the operator is updated.
	 * 
	 * @return Derivative of the operator with respect to left and right input.
	 */
	virtual std::vector<Tensor> computeDirectDerivative() const = 0;

	/**
     * @brief Returns the operator's inputs.
     * 
     */
	const std::vector<NodePtr>& getInputs() const
	{
		return inputs_;
	}

protected:
	std::vector<NodePtr> inputs_;
};

using MultipleOperatorPtr = std::shared_ptr<MultipleOperator>;

} // namespace mlCore::autoDiff::multipleOperators

#endif