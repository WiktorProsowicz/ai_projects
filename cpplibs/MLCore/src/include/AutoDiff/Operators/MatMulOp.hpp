#ifndef MLCORE_SRC_INCLUDE_AUTODIFF_OPERATORS_MATMULOP_HPP
#define MLCORE_SRC_INCLUDE_AUTODIFF_OPERATORS_MATMULOP_HPP

#include <algorithm>

#include "AutoDiff/GraphNodes.hpp"
#include "MLCore/Utilities.h"

namespace autoDiff::ops::detail
{
/**
 * @brief Performs matrix multiplication of two input nodes.
 */
class MatMulOp final : public Operator
{
public:
	MatMulOp() = delete;

	MatMulOp(const MatMulOp&) = delete;
	MatMulOp(MatMulOp&&) = delete;
	MatMulOp& operator=(const MatMulOp&) = delete;
	MatMulOp& operator=(MatMulOp&&) = delete;

	~MatMulOp() override = default;

	std::vector<size_t> getOutputShape() const override
	{
		const auto lhs = getInputs().front();
		const auto rhs = getInputs().back();

		return mlCore::detail::getOutputShapeForMatmul(lhs->getOutputShape(), rhs->getOutputShape());
	}

	const mlCore::Tensor& getValue() const override
	{
		return _value;
	}

	NodePtr copy() const
	{
		auto copiedOp = std::make_shared<MatMulOp>(getInputs());
		copiedOp->_value = _value;

		return copiedOp;
	}

	void updateValue() override
	{
		const auto lhs = getInputs().front();
		const auto rhs = getInputs().back();

		_value = lhs->getValue().matmul(rhs->getValue());
	}

	std::vector<mlCore::Tensor> computeDerivative(const mlCore::Tensor& outerDerivative) const override
	{
		const auto lhs = getInputs().front();
		const auto rhs = getInputs().back();

		std::vector<mlCore::Tensor> derivatives;
		derivatives.reserve(2);

		derivatives[0] = lhs->getValue().transposed().matmul(outerDerivative);
		derivatives[1] = outerDerivative.matmul(rhs->getValue().transposed());

		return derivatives;
	}

	std::vector<mlCore::Tensor> computeDirectDerivative() const override
	{
		const auto lhs = getInputs().front();
		const auto rhs = getInputs().back();

		std::vector<mlCore::Tensor> derivatives;
		derivatives.reserve(2);

		const mlCore::Tensor onesWithOutputShape(_value.shape(), 1.0);

		derivatives[0] = onesWithOutputShape.matmul(rhs->getValue().transposed());
		derivatives[1] = lhs->getValue().transposed().matmul(onesWithOutputShape);

		return derivatives;
	}

private:
	mlCore::Tensor _value;
};
} // namespace autoDiff::ops::detail

#endif