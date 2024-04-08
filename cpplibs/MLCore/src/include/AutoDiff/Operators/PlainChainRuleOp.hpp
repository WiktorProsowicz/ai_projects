#ifndef MLCORE_SRC_INCLUDE_AUTODIFF_OPERATORS_PLAINCHAINRULEOP_HPP
#define MLCORE_SRC_INCLUDE_AUTODIFF_OPERATORS_PLAINCHAINRULEOP_HPP

#include <algorithm>

#include "AutoDiff/Operators/ParamlessOp.hpp"

namespace autoDiff::ops::detail
{
/**
 * @brief Simple operator class, whose backward pass involves simple ultiplication of outer derivative and
 * direct derivatives.
 *
 * @details This operator is suited to nodes representing a simple function applying non-linearity
 * to the input, such as Sigmoid, or a basic operation like addition.
 */
class PlainChainRuleOp final : public ParamlessOp
{
public:
	PlainChainRuleOp() = delete;

	using ParamlessOp::ParamlessOp;

	PlainChainRuleOp(const PlainChainRuleOp&) = delete;
	PlainChainRuleOp(PlainChainRuleOp&&) = delete;
	PlainChainRuleOp& operator=(const PlainChainRuleOp&) = delete;
	PlainChainRuleOp& operator=(PlainChainRuleOp&&) = delete;

	~PlainChainRuleOp() override = default;

	/**
	 * @brief Returns a copy of the operator with the inputs and current value.
	 */
	NodePtr copy() const override
	{
		auto copiedOp = std::make_shared<PlainChainRuleOp>(getInputs(), _forwardFunc, _backwardFunc);
		copiedOp->_value = _value;

		return copiedOp;
	}

	std::vector<mlCore::Tensor> computeDerivative(const mlCore::Tensor& outerDerivative) const override
	{

		auto derivatives = computeDirectDerivative();

		std::for_each(derivatives.begin(),
					  derivatives.end(),
					  [&outerDerivative](auto& deriv) { deriv *= outerDerivative; });

		return derivatives;
	}
};
} // namespace autoDiff::ops::detail

#endif
