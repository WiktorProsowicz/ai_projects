/**********************
 * Test suite for 'ai_projects'
 * 
 * Copyright (c) 2023
 * 
 * by Wiktor Prosowicz
 **********************/

#include <AutoDiff/DerivativeExtractor.h>
#include <AutoDiff/GraphOperations.h>
#include <gtest/gtest.h>
#include <iostream>

namespace
{

template <typename OperType>
concept UnaryNodeOperation = requires(OperType oper, mlCore::NodePtr node)
{
	{
		oper(node)
		} -> std::same_as<mlCore::NodePtr>;
};

template <typename OperType>
concept BinaryNodeOperation = requires(OperType oper, const mlCore::NodePtr node)
{
	{
		oper(node, node)
		} -> std::same_as<mlCore::NodePtr>;
};

struct UnaryParams
{
	std::vector<uint64_t> tensorShape;
	std::unique_ptr<mlCore::ITensorInitializer<double>> initializer;
};

struct BinaryParams
{
	std::vector<uint64_t> leftTensorShape;
	std::unique_ptr<mlCore::ITensorInitializer<double>> leftInitializer;
	std::vector<uint64_t> rightTensorShape;
	std::unique_ptr<mlCore::ITensorInitializer<double>> rightInitializer;
};

std::string stringifyTensor(const mlCore::Tensor& tensor)
{
	std::stringstream ss;
	ss << tensor;
	return ss.str();
}

/*****************************
 * 
 * Test Fixture
 * 
 *****************************/
class TestDerivativeExtractor : public testing::Test
{
protected:
	template <UnaryNodeOperation Operation>
	mlCore::Tensor computeDefinitionDerivative(Operation oper,
											   const mlCore::Tensor& inputTensor) const
	{
		constexpr double kEpsilon = 1e-6;

		const auto backShiftedNode =
			std::make_shared<mlCore::Constant>(inputTensor - mlCore::Tensor({}, kEpsilon));
		const auto frontShiftedNode =
			std::make_shared<mlCore::Constant>(inputTensor + mlCore::Tensor({}, kEpsilon));

		const auto backShiftedResult = oper(backShiftedNode);
		const auto frontShiftedResult = oper(frontShiftedNode);

		return (frontShiftedResult->getValue() - backShiftedResult->getValue()) /
			   mlCore::Tensor({}, 2 * kEpsilon);
	}

	void compareTwoDerivatives(const mlCore::Tensor& computedDerivative,
							   const mlCore::Tensor& expDerivative,
							   const std::string& message) const
	{
		for(auto gotTensorIt = computedDerivative.begin(), expectedTensorIt = expDerivative.begin();
			(gotTensorIt < computedDerivative.end()) && (expectedTensorIt < expDerivative.end());
			gotTensorIt++, expectedTensorIt++)
		{
			ASSERT_NEAR(*gotTensorIt, *expectedTensorIt, 1e-4)
				<< message << "\n\nComputed derivative:\n"
				<< computedDerivative << "\n\nExpected derivative:\n"
				<< expDerivative;
		}
	}

	template <BinaryNodeOperation Operation>
	void testBinaryOperationDerivative(Operation oper, const BinaryParams& params) const
	{
		// input tensors
		mlCore::Tensor leftNodeValue(params.leftTensorShape);
		leftNodeValue.fill(*params.leftInitializer);

		mlCore::Tensor rightNodeValue(params.rightTensorShape);
		rightNodeValue.fill(*params.rightInitializer);

		// input graph nodes
		const auto leftInputNode = std::make_shared<mlCore::Constant>(leftNodeValue);
		const auto rightInputNode = std::make_shared<mlCore::Constant>(rightNodeValue);

		// operation result
		const auto operationResult =
			std::dynamic_pointer_cast<mlCore::IBinaryOperator>(oper(leftInputNode, rightInputNode));

		if(!operationResult)
		{
			FAIL() << "Given operation yields something different than IBinaryOperationPtr!";
		}

		const auto [leftDerivative, rightDerivative] = mlCore::DerivativeExtractor{}(
			operationResult, mlCore::Tensor(operationResult->getValue().shape(), 1.0));

		// operation with locked right input
		const auto leftLockedOperation = [&oper, &rightInputNode](mlCore::NodePtr node) {
			return oper(node, rightInputNode);
		};
		// operation with locked left input
		const auto rightLockedOperation = [&oper, &leftInputNode](mlCore::NodePtr node) {
			return oper(leftInputNode, node);
		};

		// computing derivative according to definition
		const auto leftDefDerivative =
			computeDefinitionDerivative(leftLockedOperation, leftNodeValue);

		const auto rightDefDerivative =
			computeDefinitionDerivative(rightLockedOperation, rightNodeValue);

		compareTwoDerivatives(leftDerivative,
							  leftDefDerivative,
							  "Found inequality while comparing DerivativeExtractor result and "
							  "definition derivative with regard to left input!\n\nInputs:\n\n" +
								  stringifyTensor(leftNodeValue) + "\n\n" +
								  stringifyTensor(rightNodeValue));

		compareTwoDerivatives(rightDerivative,
							  rightDefDerivative,
							  "Found inequality while comparing DerivativeExtractor result and "
							  "definition derivative with regard to right input!\n\nInputs:\n\n" +
								  stringifyTensor(leftNodeValue) + "\n\n" +
								  stringifyTensor(rightNodeValue));
	}

	template <UnaryNodeOperation Operation>
	void testUnaryOperationDerivative(Operation oper, const UnaryParams& params) const
	{
		// input tensor
		mlCore::Tensor nodeValue(params.tensorShape);
		nodeValue.fill(*params.initializer);

		// input graph nodes
		const auto inputNode = std::make_shared<mlCore::Constant>(nodeValue);

		// operation result
		const auto operationResult =
			std::dynamic_pointer_cast<mlCore::IUnaryOperator>(oper(inputNode));

		if(!operationResult)
		{
			FAIL() << "Given operation yields something different than IUnaryOperationPtr!";
		}

		const auto derivative = mlCore::DerivativeExtractor{}(
			operationResult, mlCore::Tensor(operationResult->getValue().shape(), 1.0));

		// computing derivative according to definition
		const auto definitionDerivative = computeDefinitionDerivative(oper, nodeValue);

		compareTwoDerivatives(
			derivative,
			definitionDerivative,
			"Found inequality while comparing derivative, computed with "
			"DerivativeExtractor, and that computed according to definition!\n\nInput node:\n\n" +
				stringifyTensor(nodeValue));
	}

	void
	testMatmulDerivativeExtracting(const mlCore::Tensor& leftTensor,
								   const mlCore::Tensor& rightTensor,
								   const mlCore::Tensor& outer,
								   const std::pair<mlCore::Tensor, mlCore::Tensor>& expected) const
	{
		const auto leftNode = std::make_shared<mlCore::Constant>(leftTensor);
		const auto rightNode = std::make_shared<mlCore::Constant>(rightTensor);

		const auto operationResult = std::dynamic_pointer_cast<mlCore::IBinaryOperator>(
			mlCore::BinaryOperations{}.matmul(leftNode, rightNode));

		if(!operationResult)
		{
			FAIL() << "Matmul operation yielded something other than IBinaryOperator!";
		}

		const auto [leftDerivative, rightDerivative] =
			mlCore::DerivativeExtractor{}(operationResult, outer);

		const auto& [leftExpected, rightExpected] = expected;

		std::stringstream messageGenerator;
		messageGenerator << "This test needs manual check:\n\nInputs:\n\n"
						 << leftTensor << "\n\n"
						 << rightTensor << "\n\nOuter derivative:\n\n"
						 << outer;

		compareTwoDerivatives(leftDerivative, leftExpected, messageGenerator.str());

		compareTwoDerivatives(rightDerivative, rightExpected, messageGenerator.str());
	}
};

/*****************************
 * 
 * Particular test calls
 * 
 *****************************/

TEST_F(TestDerivativeExtractor, testReluDerivative)
{
	const UnaryParams params{
		.tensorShape = {3, 3},
		.initializer = std::make_unique<mlCore::RangeTensorInitializer<double>>(-3.0, .7)};

	testUnaryOperationDerivative(
		[](mlCore::NodePtr node) { return mlCore::NodesActivations{}.relu(node); }, params);
}

TEST_F(TestDerivativeExtractor, testSigmoidDerivative)
{
	const UnaryParams params{
		.tensorShape = {5, 5},
		.initializer = std::make_unique<mlCore::RangeTensorInitializer<double>>(-9.0, .7)};

	testUnaryOperationDerivative(
		[](mlCore::NodePtr node) { return mlCore::NodesActivations{}.sigmoid(node); }, params);
}

TEST_F(TestDerivativeExtractor, testMultiplyDerivative)
{
	const BinaryParams params{
		.leftTensorShape = {3, 5},
		.leftInitializer = std::make_unique<mlCore::RangeTensorInitializer<double>>(-4, .8),
		.rightTensorShape = {3, 5},
		.rightInitializer = std::make_unique<mlCore::RangeTensorInitializer<double>>(5, .3)};

	testBinaryOperationDerivative(
		[](mlCore::NodePtr left, mlCore::NodePtr right) {
			return mlCore::BinaryOperations{}.multiply(left, right);
		},
		params);
}

TEST_F(TestDerivativeExtractor, testDivideDerivative)
{
	const BinaryParams params{
		.leftTensorShape = {3, 5},
		.leftInitializer = std::make_unique<mlCore::RangeTensorInitializer<double>>(-4, .7),
		.rightTensorShape = {3, 5},
		.rightInitializer = std::make_unique<mlCore::RangeTensorInitializer<double>>(5, .3)};

	testBinaryOperationDerivative(
		[](mlCore::NodePtr left, mlCore::NodePtr right) {
			return mlCore::BinaryOperations{}.divide(left, right);
		},
		params);
}

TEST_F(TestDerivativeExtractor, testAddDerivative)
{
	const BinaryParams params{
		.leftTensorShape = {3, 5},
		.leftInitializer = std::make_unique<mlCore::RangeTensorInitializer<double>>(-4, .7),
		.rightTensorShape = {3, 5},
		.rightInitializer = std::make_unique<mlCore::RangeTensorInitializer<double>>(5, .3)};

	testBinaryOperationDerivative(
		[](mlCore::NodePtr left, mlCore::NodePtr right) {
			return mlCore::BinaryOperations{}.add(left, right);
		},
		params);
}

TEST_F(TestDerivativeExtractor, testSubtractDerivative)
{
	const BinaryParams params{
		.leftTensorShape = {3, 5},
		.leftInitializer = std::make_unique<mlCore::RangeTensorInitializer<double>>(-4, .7),
		.rightTensorShape = {3, 5},
		.rightInitializer = std::make_unique<mlCore::RangeTensorInitializer<double>>(5, .3)};

	testBinaryOperationDerivative(
		[](mlCore::NodePtr left, mlCore::NodePtr right) {
			return mlCore::BinaryOperations{}.subtract(left, right);
		},
		params);
}

TEST_F(TestDerivativeExtractor, testPowerDerivative)
{
	const auto powerLambda = [](mlCore::NodePtr left, mlCore::NodePtr right) {
		return mlCore::BinaryOperations{}.power(left, right);
	};

	const BinaryParams paramsWithLeftScalar{
		.leftTensorShape = {3, 5},
		.leftInitializer = std::make_unique<mlCore::RangeTensorInitializer<double>>(1.3, .1),
		.rightTensorShape = {3, 5},
		.rightInitializer = std::make_unique<mlCore::RangeTensorInitializer<double>>(5, .1)};

	testBinaryOperationDerivative(powerLambda, paramsWithLeftScalar);
}

TEST_F(TestDerivativeExtractor, testMatmulDerivative)
{
	mlCore::Tensor leftInput({4, 3});
	leftInput.fill(mlCore::RangeTensorInitializer<double>(-4, .7));

	mlCore::Tensor rightInput({3, 2});
	rightInput.fill(mlCore::RangeTensorInitializer<double>(5, .3));

	mlCore::Tensor outerDerivative({4, 2});
	outerDerivative.fill({1, 2, 3, 4}, true);

	mlCore::Tensor leftExpected({4, 3});
	leftExpected.fill({15.6, 17.4, 19.2, 36.2, 40.4, 44.6, 15.6, 17.4, 19.2, 36.2, 40.4, 44.6});

	mlCore::Tensor rightExpected({3, 2});
	rightExpected.fill({-2.6, -6, 3, 2.4, 8.6, 10.8});

	testMatmulDerivativeExtracting(
		leftInput, rightInput, outerDerivative, std::pair{leftExpected, rightExpected});
}

} // namespace
