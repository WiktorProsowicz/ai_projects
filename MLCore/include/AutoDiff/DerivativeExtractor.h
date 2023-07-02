#ifndef AUTODIFF_DERIVATIVEEXTRACTOR_H
#define AUTODIFF_DERIVATIVEEXTRACTOR_H

#include <AutoDiff/UnaryOperators/ReluOperator.h>
#include <AutoDiff/UnaryOperators/SigmoidOperator.h>

#include <AutoDiff/BinaryOperators/AddOperator.h>
#include <AutoDiff/BinaryOperators/DivideOperator.h>
#include <AutoDiff/BinaryOperators/MatmulOperator.h>
#include <AutoDiff/BinaryOperators/MultiplyOperator.h>
#include <AutoDiff/BinaryOperators/PowerOperator.h>
#include <AutoDiff/BinaryOperators/SubtractOperator.h>

namespace mlCore
{
/**
 * @brief Defines ways to compute derivative based on the operator's type and values of its inputs. 
 * 
 * I.e let @arg oper be f(x, y)
 * DerivativeExtractor::operator(f(x, y)) -> {df/dx, df/dy} 
 * 
 */
class DerivativeExtractor
{
public:
	DerivativeExtractor() = default;

	DerivativeExtractor(const DerivativeExtractor&) = delete; /// Copy constructor
	DerivativeExtractor(DerivativeExtractor&&) = delete; /// Move constructor
	DerivativeExtractor& operator=(const DerivativeExtractor&) = delete; /// Copy assignment
	DerivativeExtractor& operator=(DerivativeExtractor&&) = delete; /// Move assignment

	/**
     * @brief Computes derivative of an unary operator in respect of its input
     * 
     * @param oper operator to compute derivative from
     * @param outerDerivative derivative to perform chain rule computation
     * @return derivative of oper updated with outerDerivative
     */
	Tensor operator()(const IUnaryOperatorPtr oper, const Tensor& outerDerivative) const;

	/**
     * @brief Computes derivatives of a binary operator in respect of its inputs
     * 
     * @param oper operator to compute derivative from
     * @param outerDerivative derivative to perform chain rule computation
     * @return derivatives of oper updated with outerDerivative
     */
	std::pair<Tensor, Tensor> operator()(const IBinaryOperatorPtr oper,
										 const Tensor& outerDerivative) const;

private:
	/// Extraction functions for concrete unary operators' subclasses
	Tensor extract(const ReluOperatorPtr oper, const Tensor& outerDerivative) const;
	Tensor extract(const SigmoidOperatorPtr oper, const Tensor& outerDerivative) const;

	/// Extraction functions for concrete binary operators' subclasses
	std::pair<Tensor, Tensor> extract(const AddOperatorPtr oper,
									  const Tensor& outerDerivative) const;
	std::pair<Tensor, Tensor> extract(const DivideOperatorPtr oper,
									  const Tensor& outerDerivative) const;
	std::pair<Tensor, Tensor> extract(const MatmulOperatorPtr oper,
									  const Tensor& outerDerivative) const;
	std::pair<Tensor, Tensor> extract(const MultiplyOperatorPtr oper,
									  const Tensor& outerDerivative) const;
	std::pair<Tensor, Tensor> extract(const PowerOperatorPtr oper,
									  const Tensor& outerDerivative) const;
	std::pair<Tensor, Tensor> extract(const SubtractOperatorPtr oper,
									  const Tensor& outerDerivative) const;
};
} // namespace mlCore

#endif