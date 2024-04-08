#ifndef MLCORE_SRC_INCLUDE_AUTODIFF_OPERATORS_STATELESSOP_HPP
#define MLCORE_SRC_INCLUDE_AUTODIFF_OPERATORS_STATELESSOP_HPP

#include <functional>
#include <vector>

#include <LoggingLib/LoggingLib.hpp>
#include <fmt/core.h>

#include "AutoDiff/GraphNodes.hpp"
#include "MLCore/BasicTensor.h"

namespace autoDiff::ops::detail
{
using ForwardFunction = std::function<mlCore::Tensor(const std::vector<autoDiff::NodePtr>&)>;
using BackwardFunction = std::function<std::vector<mlCore::Tensor>(const std::vector<autoDiff::NodePtr>&)>;

/**
 * @brief Represents an operator that does not preserve any additional parameters.
 *
 * It is performed via functions given at construction. This class is not complete, therefore a specific
 * extension point should be defined that would override the `copy` and `computeDerivative` methods.
 */
class ParamlessOp : public Operator
{
public:
	ParamlessOp() = delete;

	/**
	 * @brief Creates the op setting the functions it shall use.
	 * @param inputs See autoDiff::Operator.
	 * @param forwardFunc Function used to update the internal value depending on the values of the inputs.
	 * @param backwardFunc Function used to compute direct derivatives.
	 */
	ParamlessOp(std::vector<NodePtr> inputs, ForwardFunction forwardFunc, BackwardFunction backwardFunc)
		: Operator(std::move(inputs))
		, _forwardFunc(std::move(forwardFunc))
		, _backwardFunc(std::move(backwardFunc))
		, _value(getInputs().front()->getOutputShape())
	{}

	ParamlessOp(const ParamlessOp&) = delete;
	ParamlessOp(ParamlessOp&&) = delete;
	ParamlessOp& operator=(const ParamlessOp&) = delete;
	ParamlessOp& operator=(ParamlessOp&&) = delete;

	~ParamlessOp() override = default;

	const mlCore::Tensor& getValue() const override
	{
		return _value;
	}

	const std::vector<size_t>& getOutputShape() const override
	{
		return getInputs().front()->getOutputShape();
	}

	void updateValue() override
	{
		_value = _forwardFunc(getInputs());
	}

	std::vector<mlCore::Tensor> computeDirectDerivative() const override
	{
		return _backwardFunc(getInputs());
	}

protected:
	ForwardFunction _forwardFunc;
	BackwardFunction _backwardFunc;
	mlCore::Tensor _value;
};
} // namespace autoDiff::ops::detail

#endif
