#include <AutoDiff/DerivativeExtractor.h>
#include <MLCore/TensorOperations.h>

namespace mlCore
{
Tensor DerivativeExtractor::operator()(const IUnaryOperatorPtr oper,
									   const Tensor& outerDerivative) const
{
	if(const auto casted = std::dynamic_pointer_cast<ReluOperator>(oper))
		return extract(casted, outerDerivative);

	else if(const auto casted = std::dynamic_pointer_cast<SigmoidOperator>(oper))
		return extract(casted, outerDerivative);

	LOG_ERROR("DerivativeExtractor", "Unknown type of unary operator - can't compute derivative.")
	return {{}, 0};
}

std::pair<Tensor, Tensor> DerivativeExtractor::operator()(const IBinaryOperatorPtr oper,
														  const Tensor& outerDerivative) const
{
	if(const auto casted = std::dynamic_pointer_cast<AddOperator>(oper))
		return extract(casted, outerDerivative);

	else if(const auto casted = std::dynamic_pointer_cast<DivideOperator>(oper))
		return extract(casted, outerDerivative);

	else if(const auto casted = std::dynamic_pointer_cast<MatmulOperator>(oper))
		return extract(casted, outerDerivative);

	else if(const auto casted = std::dynamic_pointer_cast<MultiplyOperator>(oper))
		return extract(casted, outerDerivative);

	else if(const auto casted = std::dynamic_pointer_cast<PowerOperator>(oper))
		return extract(casted, outerDerivative);

	else if(const auto casted = std::dynamic_pointer_cast<SubtractOperator>(oper))
		return extract(casted, outerDerivative);

	LOG_ERROR("DerivativeExtractor", "Unknown type of binary operator - can't compute derivatives.")
	return {{{}, 0}, {{}, 0}};
}

Tensor DerivativeExtractor::extract(const ReluOperatorPtr oper, const Tensor& outerDerivative) const
{
	Tensor inputCopy = oper->getInput()->getValue();
	for(auto& val : inputCopy)
		val = val > 0 ? 1 : 0;

	return inputCopy * outerDerivative;
}

Tensor DerivativeExtractor::extract(const SigmoidOperatorPtr oper,
									const Tensor& outerDerivative) const
{
	Tensor inputCopy = oper->getInput()->getValue();
	for(auto& val : inputCopy)
		val = pow(M_E, val) * (1 - pow(M_E, val));

	return inputCopy * outerDerivative;
}

std::pair<Tensor, Tensor> DerivativeExtractor::extract(const AddOperatorPtr oper,
													   const Tensor& outerDerivative) const
{
	return {outerDerivative, outerDerivative};
}

std::pair<Tensor, Tensor> DerivativeExtractor::extract(const SubtractOperatorPtr oper,
													   const Tensor& outerDerivative) const
{
	return {outerDerivative, -outerDerivative};
}

std::pair<Tensor, Tensor> DerivativeExtractor::extract(const DivideOperatorPtr oper,
													   const Tensor& outerDerivative) const
{
	const auto& [leftInputNode, rightInputNode] = oper->getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	return {(Tensor(rightValue.shape(), 1) / rightValue) * outerDerivative,
			-leftValue / (rightValue * rightValue) * outerDerivative};
}

std::pair<Tensor, Tensor> DerivativeExtractor::extract(const MatmulOperatorPtr oper,
													   const Tensor& outerDerivative) const
{
	const auto& [leftInputNode, rightInputNode] = oper->getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	return {outerDerivative.matmul(rightValue.transposed()),
			leftValue.transposed().matmul(outerDerivative)};
}

std::pair<Tensor, Tensor> DerivativeExtractor::extract(const MultiplyOperatorPtr oper,
													   const Tensor& outerDerivative) const
{
	const auto& [leftInputNode, rightInputNode] = oper->getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	return {rightValue * outerDerivative, leftValue * outerDerivative};
}

std::pair<Tensor, Tensor> DerivativeExtractor::extract(const PowerOperatorPtr oper,
													   const Tensor& outerDerivative) const
{
	const auto& [leftInputNode, rightInputNode] = oper->getInputs();
	const auto& leftValue = leftInputNode->getValue();
	const auto& rightValue = rightInputNode->getValue();

	return {TensorOperations::power(leftValue, rightValue - Tensor(rightValue.shape(), 1)) *
				outerDerivative * rightValue,
			TensorOperations::ln(leftValue) * TensorOperations::power(leftValue, rightValue) *
				outerDerivative};
}
} // namespace mlCore