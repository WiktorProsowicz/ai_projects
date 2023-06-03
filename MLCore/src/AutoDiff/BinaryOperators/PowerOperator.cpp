#include <AutoDiff/BinaryOperators/PowerOperator.h>
#include <MLCore/TensorOperations.h>

namespace mlCore
{
void PowerOperator::updateValue()
{
	if(const auto castedLeft = std::dynamic_pointer_cast<Constant>(lhsInput_);
	   lhsInput_ && (lhsInput_->getValue().nDimensions() == 0))
	{
		value_ = TensorOperations::power(lhsInput_->getValue(), rhsInput_->getValue());
	}
	else if(const auto castedRight = std::dynamic_pointer_cast<Constant>(rhsInput_);
			rhsInput_ && (rhsInput_->getValue().nDimensions() == 0))
	{
		value_ = TensorOperations::power(lhsInput_->getValue(), rhsInput_->getValue());
	}
	else
	{
		LOG_ERROR("BinaryOperator",
				  "For node " << name_
							  << " of type 'POWER' at least one of the operator sides has to "
								 "be a scalar constant.");
	}
}
} // namespace mlCore