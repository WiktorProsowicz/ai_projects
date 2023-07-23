#include <AutoDiff/DerivativeExtractor.h>
#include <MLCore/TensorOperations.h>

namespace mlCore::autoDiff
{
Tensor DerivativeExtractor::operator()(const unaryOperators::UnaryOperatorPtr oper,
									   const Tensor& outerDerivative) const
{
	if(const auto casted = std::dynamic_pointer_cast<unaryOperators::ReluOperator>(oper))
	{
		return extract(casted, outerDerivative);
	}

	if(const auto casted = std::dynamic_pointer_cast<unaryOperators::SigmoidOperator>(oper))
	{
		return extract(casted, outerDerivative);
	}

	LOG_ERROR("DerivativeExtractor", "Unknown type of unary operator - can't compute derivative.")
	return {{}, 0};
}

std::pair<Tensor, Tensor>
DerivativeExtractor::operator()(const binaryOperators::BinaryOperatorPtr oper,
								const Tensor& outerDerivative) const
{
	if(const auto casted = std::dynamic_pointer_cast<binaryOperators::AddOperator>(oper))
	{
		return extract(casted, outerDerivative);
	}
	if(const auto casted = std::dynamic_pointer_cast<binaryOperators::DivideOperator>(oper))
	{
		return extract(casted, outerDerivative);
	}
	if(const auto casted = std::dynamic_pointer_cast<binaryOperators::MatmulOperator>(oper))
	{
		return extract(casted, outerDerivative);
	}
	if(const auto casted = std::dynamic_pointer_cast<binaryOperators::MultiplyOperator>(oper))
	{
		return extract(casted, outerDerivative);
	}
	if(const auto casted = std::dynamic_pointer_cast<binaryOperators::PowerOperator>(oper))
	{
		return extract(casted, outerDerivative);
	}
	if(const auto casted = std::dynamic_pointer_cast<binaryOperators::SubtractOperator>(oper))
	{
		return extract(casted, outerDerivative);
	}

	LOG_ERROR("DerivativeExtractor", "Unknown type of binary operator - can't compute derivatives.")
	return {{{}, 0}, {{}, 0}};
}

Tensor DerivativeExtractor::extract(const unaryOperators::ReluOperatorPtr oper,
									const Tensor& outerDerivative)
{
	Tensor inputCopy = oper->getValue();
	for(auto& val : inputCopy)
	{
		val = val > 0 ? 1 : 0;
	}

	return inputCopy * outerDerivative;
}

Tensor DerivativeExtractor::extract(const unaryOperators::SigmoidOperatorPtr oper,
									const Tensor& outerDerivative)
{
	Tensor inputCopy = oper->getValue();

	for(auto& val : inputCopy)
	{
		val = val * (1 - val);
	}

	return inputCopy * outerDerivative;
}

std::pair<Tensor, Tensor>
DerivativeExtractor::extract(const binaryOperators::AddOperatorPtr /*unused*/,
							 const Tensor& outerDerivative)
{
	return {outerDerivative, outerDerivative};
}

std::pair<Tensor, Tensor>
DerivativeExtractor::extract(const binaryOperators::SubtractOperatorPtr /*unused*/,
							 const Tensor& outerDerivative)
{
	return {outerDerivative, -outerDerivative};
}

std::pair<Tensor, Tensor>
DerivativeExtractor::extract(const binaryOperators::DivideOperatorPtr oper,
							 const Tensor& outerDerivative)
{
	const auto& [leftInputNode, rightInputNode] = oper->getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	return {(Tensor(rightValue.shape(), 1) / rightValue) * outerDerivative,
			-leftValue / (rightValue * rightValue) * outerDerivative};
}

std::pair<Tensor, Tensor>
DerivativeExtractor::extract(const binaryOperators::MatmulOperatorPtr oper,
							 const Tensor& outerDerivative)
{
	const auto& [leftInputNode, rightInputNode] = oper->getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	return {outerDerivative.matmul(rightValue.transposed()),
			leftValue.transposed().matmul(outerDerivative)};
}

std::pair<Tensor, Tensor>
DerivativeExtractor::extract(const binaryOperators::MultiplyOperatorPtr oper,
							 const Tensor& outerDerivative)
{
	const auto& [leftInputNode, rightInputNode] = oper->getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	return {rightValue * outerDerivative, leftValue * outerDerivative};
}

std::pair<Tensor, Tensor> DerivativeExtractor::extract(const binaryOperators::PowerOperatorPtr oper,
													   const Tensor& outerDerivative)
{
	const auto& [leftInputNode, rightInputNode] = oper->getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	return {TensorOperations::power(leftValue, rightValue - Tensor(rightValue.shape(), 1)) *
				outerDerivative * rightValue,
			TensorOperations::ln(leftValue) * TensorOperations::power(leftValue, rightValue) *
				outerDerivative};
}
} // namespace mlCore::autoDiff