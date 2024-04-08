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

	explicit MatMulOp(const std::vector<NodePtr>& inputs)
		: Operator(inputs)
		, _outputShape(_computeOutputShapeSafe())
		, _value(_outputShape)
	{}

	MatMulOp(const MatMulOp&) = delete;
	MatMulOp(MatMulOp&&) = delete;
	MatMulOp& operator=(const MatMulOp&) = delete;
	MatMulOp& operator=(MatMulOp&&) = delete;

	~MatMulOp() override = default;

	const std::vector<size_t>& getOutputShape() const override
	{
		return _outputShape;
	}

	const mlCore::Tensor& getValue() const override
	{
		return _value;
	}

	NodePtr copy() const override
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

		derivatives.emplace_back(outerDerivative.matmul(rhs->getValue().transposed()));
		derivatives.emplace_back(lhs->getValue().transposed().matmul(outerDerivative));

		return derivatives;
	}

	std::vector<mlCore::Tensor> computeDirectDerivative() const override
	{
		const auto lhs = getInputs().front();
		const auto rhs = getInputs().back();

		std::vector<mlCore::Tensor> derivatives;
		derivatives.reserve(2);

		const mlCore::Tensor onesWithOutputShape(_value.shape(), 1.0);

		derivatives.emplace_back(onesWithOutputShape.matmul(rhs->getValue().transposed()));
		derivatives.emplace_back(lhs->getValue().transposed().matmul(onesWithOutputShape));

		return derivatives;
	}

private:
	std::vector<size_t> _computeOutputShapeSafe() const noexcept
	{
		return mlCore::detail::getOutputShapeForMatmul(getInputs().front()->getOutputShape(),
													   getInputs().back()->getOutputShape());
	}

	std::vector<size_t> _outputShape;
	mlCore::Tensor _value;
};
} // namespace autoDiff::ops::detail

#endif
