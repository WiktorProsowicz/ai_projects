#ifndef MLCORE_SRC_INCLUDE_AUTODIFF_OPERATORS_MATMULOP_HPP
#define MLCORE_SRC_INCLUDE_AUTODIFF_OPERATORS_MATMULOP_HPP

#include <algorithm>

#include "AutoDiff/GraphNodes.hpp"
#include "MLCore/TensorOperations.h"
#include "MLCore/Utilities.h"
#include "MLCore/UtilitiesImpl.h"

namespace autoDiff::ops::detail
{
/**
 * @brief Performs matrix multiplication of two input nodes.
 */
class MatMulOp final : public Operator
{
public:
	MatMulOp() = delete;

	/**
	 * @brief MatMulOp constructor.
	 *
	 * @param inputs Input nodes (expected two nodes).
	 * @param lhsSpec Specification for the left-hand side matrix.
	 * @param rhsSpec Specification for the right-hand side matrix.
	 * @param avoidMatrixOutput If true and if the output tensor is either a row or a column vector, the '1'
	 * dimension shall be trimmed.
	 */
	explicit MatMulOp(const std::vector<NodePtr>& inputs,
					  mlCore::MatrixSpec lhsSpec = mlCore::MatrixSpec::Default,
					  mlCore::MatrixSpec rhsSpec = mlCore::MatrixSpec::Default,
					  bool avoidMatrixOutput = false)
		: Operator(inputs)
		, _finalOutputShape(_computeFinalOutputShape())
		, _originalOutputShape(_computeOriginalOutputShape())
		, _value(_finalOutputShape)
		, _lhsSpec(lhsSpec)
		, _rhsSpec(rhsSpec)
		, _avoidMatrixOutput(avoidMatrixOutput)
	{}

	MatMulOp(const MatMulOp&) = delete;
	MatMulOp(MatMulOp&&) = delete;
	MatMulOp& operator=(const MatMulOp&) = delete;
	MatMulOp& operator=(MatMulOp&&) = delete;

	~MatMulOp() override = default;

	const std::vector<size_t>& getOutputShape() const override
	{
		return _finalOutputShape;
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

		_value = mlCore::TensorOperations::matmul(lhs->getValue(), rhs->getValue(), _lhsSpec, _rhsSpec);

		if(_originalOutputShape != _finalOutputShape)
		{
			_value.reshape(mlCore::detail::trimRowOrColumnVector(_value.shape()));
		}
	}

	std::vector<mlCore::Tensor> computeDerivative(const mlCore::Tensor& outerDerivative) const override
	{
		return computeDerivativeUniversal(outerDerivative, _getOuterDerivativeSpec());
	}

	std::vector<mlCore::Tensor> computeDirectDerivative() const override
	{
		return computeDerivativeUniversal(mlCore::Tensor(_originalOutputShape, 1.0),
										  mlCore::MatrixSpec::Default);
	}

private:
	/// @brief Computes output shape for matrix multiplication, taking into account the matrix specifications.
	std::vector<size_t> _computeFinalOutputShape() const noexcept
	{
		auto outputShape = _computeOriginalOutputShape();

		if(_avoidMatrixOutput && mlCore::detail::isRowOrColumnVector(outputShape))
		{
			return mlCore::detail::trimRowOrColumnVector(outputShape);
		}

		return outputShape;
	}

	/// @brief Computes the raw output shape of the matrix multiplication.
	std::vector<size_t> _computeOriginalOutputShape() const noexcept
	{
		using mlCore::detail::applyMatSpecToShape;

		const auto& lhs = getInputs().front();
		const auto& rhs = getInputs().back();

		auto lhsShape = applyMatSpecToShape(lhs->getOutputShape(), _lhsSpec);
		auto rhsShape = applyMatSpecToShape(rhs->getOutputShape(), _rhsSpec);

		return mlCore::detail::getOutputShapeForMatmul(lhsShape, rhsShape);
	}

	mlCore::MatrixSpec _getOuterDerivativeSpec() const
	{
		if(_originalOutputShape != _finalOutputShape)
		{
			if(_originalOutputShape[_originalOutputShape.size() - 1] == 1)
			{
				return mlCore::MatrixSpec::ColumnVector;
			}

			return mlCore::MatrixSpec::RowVector;
		}

		return mlCore::MatrixSpec::Default;
	}

	/// @brief Computes derivative in universal case where the `outerDerivative` is either the real outer
	/// derivative or a '1's tensor.
	std::vector<mlCore::Tensor> computeDerivativeUniversal(const mlCore::Tensor& outerDerivative,
														   const mlCore::MatrixSpec outerDerivativeSpec) const
	{
		using TensorOps = mlCore::TensorOperations;
		using MatSpec = mlCore::MatrixSpec;

		const auto lhs = getInputs().front();
		const auto rhs = getInputs().back();

		auto lhsDerivative = TensorOps::matmul(outerDerivative,
											   TensorOps::transpose(rhs->getValue(), _rhsSpec),
											   outerDerivativeSpec,
											   MatSpec::Default);

		auto rhsDerivative = TensorOps::matmul(TensorOps::transpose(lhs->getValue(), _lhsSpec),
											   outerDerivative,
											   MatSpec::Default,
											   outerDerivativeSpec);

		// Adjusting the shape of the derivatives to match the inputs' shapes.
		{
			if(_lhsSpec != mlCore::MatrixSpec::Default)
			{
				lhsDerivative.reshape(lhs->getOutputShape());
			}

			if(_rhsSpec != mlCore::MatrixSpec::Default)
			{
				rhsDerivative.reshape(rhs->getOutputShape());
			}
		}

		return {std::move(lhsDerivative), std::move(rhsDerivative)};
	}

	std::vector<size_t> _finalOutputShape;
	std::vector<size_t> _originalOutputShape;
	mlCore::Tensor _value;
	mlCore::MatrixSpec _lhsSpec;
	mlCore::MatrixSpec _rhsSpec;
	bool _avoidMatrixOutput;
};
} // namespace autoDiff::ops::detail

#endif
