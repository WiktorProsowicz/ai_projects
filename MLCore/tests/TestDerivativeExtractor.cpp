/**********************
 * Test suite for 'ai_projects'
 * 
 * Copyright (c) 2023
 * 
 * by Wiktor Prosowicz
 **********************/

#include <AutoDiff/GraphOperations.h>
#include <AutoDiff/BinaryOperators/BinaryOperator.h>
#include <AutoDiff/UnaryOperators/UnaryOperator.h>

#include <gtest/gtest.h>
#include <iostream>

#include <MLCore/TensorInitializers/RangeTensorInitializer.h>
namespace
{

template <typename OperType>
concept UnaryNodeOperation = requires(OperType oper, mlCore::autoDiff::NodePtr node)
{
	{
		oper(node)
		} -> std::same_as<mlCore::autoDiff::NodePtr>;
};

template <typename OperType>
concept BinaryNodeOperation = requires(OperType oper, const mlCore::autoDiff::NodePtr node)
{
	{
		oper(node, node)
		} -> std::same_as<mlCore::autoDiff::NodePtr>;
};

struct UnaryParams
{
	std::vector<uint64_t> tensorShape;
	std::unique_ptr<mlCore::tensorInitializers::ITensorInitializer<double>> initializer;
};

struct BinaryParams
{
	std::vector<uint64_t> leftTensorShape;
	std::unique_ptr<mlCore::tensorInitializers::ITensorInitializer<double>> leftInitializer;
	std::vector<uint64_t> rightTensorShape;
	std::unique_ptr<mlCore::tensorInitializers::ITensorInitializer<double>> rightInitializer;
};

std::string stringifyTensor(const mlCore::Tensor& tensor)
{
	std::stringstream serializeStream;
	serializeStream << tensor;
	return serializeStream.str();
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
	static mlCore::Tensor computeDefinitionDerivative(Operation oper, const mlCore::Tensor& inputTensor)
	{
		constexpr double kEpsilon = 1e-6;

		const auto backShiftedNode = std::make_shared<mlCore::autoDiff::Constant>(inputTensor - mlCore::Tensor({}, kEpsilon));
		const auto frontShiftedNode = std::make_shared<mlCore::autoDiff::Constant>(inputTensor + mlCore::Tensor({}, kEpsilon));

		const auto backShiftedResult = oper(backShiftedNode);
		const auto frontShiftedResult = oper(frontShiftedNode);

		return (frontShiftedResult->getValue() - backShiftedResult->getValue()) / mlCore::Tensor({}, 2 * kEpsilon);
	}

	static void compareTwoDerivatives(const mlCore::Tensor& computedDerivative,
									  const mlCore::Tensor& expDerivative,
									  const std::string& message)
	{
		for(auto gotTensorIt = computedDerivative.begin(), expectedTensorIt = expDerivative.begin();
			(gotTensorIt < computedDerivative.end()) && (expectedTensorIt < expDerivative.end());
			gotTensorIt++, expectedTensorIt++)
		{
			ASSERT_NEAR(*gotTensorIt, *expectedTensorIt, 1e-4) << message << "\n\nComputed derivative:\n"
															   << computedDerivative << "\n\nExpected derivative:\n"
															   << expDerivative;
		}
	}

	template <BinaryNodeOperation Operation>
	static void testBinaryOperationDerivative(Operation oper, const BinaryParams& params)
	{
		using mlCore::autoDiff::binaryOperators::BinaryOperator;

		// input tensors
		mlCore::Tensor leftNodeValue(params.leftTensorShape);
		leftNodeValue.fill(*params.leftInitializer);

		mlCore::Tensor rightNodeValue(params.rightTensorShape);
		rightNodeValue.fill(*params.rightInitializer);

		// input graph nodes
		const auto leftInputNode = std::make_shared<mlCore::autoDiff::Constant>(leftNodeValue);
		const auto rightInputNode = std::make_shared<mlCore::autoDiff::Constant>(rightNodeValue);

		// operation result
		auto operationResult = std::dynamic_pointer_cast<BinaryOperator>(oper(leftInputNode, rightInputNode));

		if(!operationResult)
		{
			FAIL() << "Given operation yields something different than IBinaryOperationPtr!";
		}

		const auto [leftDerivative, rightDerivative] = operationResult->computeDirectDerivative();

		// operation with locked right input
		const auto leftLockedOperation = [&oper, &rightInputNode](mlCore::autoDiff::NodePtr node) {
			return oper(node, rightInputNode);
		};
		// operation with locked left input
		const auto rightLockedOperation = [&oper, &leftInputNode](mlCore::autoDiff::NodePtr node) {
			return oper(leftInputNode, node);
		};

		// computing derivative according to definition
		const auto leftDefDerivative = computeDefinitionDerivative(leftLockedOperation, leftNodeValue);

		const auto rightDefDerivative = computeDefinitionDerivative(rightLockedOperation, rightNodeValue);

		compareTwoDerivatives(leftDerivative,
							  leftDefDerivative,
							  "Found inequality while comparing DerivativeExtractor result and "
							  "definition derivative with regard to left input!\n\nInputs:\n\n" +
								  stringifyTensor(leftNodeValue) + "\n\n" + stringifyTensor(rightNodeValue));

		compareTwoDerivatives(rightDerivative,
							  rightDefDerivative,
							  "Found inequality while comparing DerivativeExtractor result and "
							  "definition derivative with regard to right input!\n\nInputs:\n\n" +
								  stringifyTensor(leftNodeValue) + "\n\n" + stringifyTensor(rightNodeValue));
	}

	template <UnaryNodeOperation Operation>
	static void testUnaryOperationDerivative(Operation oper, const UnaryParams& params)
	{
		using mlCore::autoDiff::unaryOperators::UnaryOperator;

		// input tensor
		mlCore::Tensor nodeValue(params.tensorShape);
		nodeValue.fill(*params.initializer);

		// input graph nodes
		const auto inputNode = std::make_shared<mlCore::autoDiff::Constant>(nodeValue);

		// operation result
		const auto operationResult = std::dynamic_pointer_cast<UnaryOperator>(oper(inputNode));

		if(!operationResult)
		{
			FAIL() << "Given operation yields something different than IUnaryOperationPtr!";
		}

		const auto derivative = operationResult->computeDirectDerivative();

		// computing derivative according to definition
		const auto definitionDerivative = computeDefinitionDerivative(oper, nodeValue);

		compareTwoDerivatives(derivative,
							  definitionDerivative,
							  "Found inequality while comparing derivative, computed with "
							  "DerivativeExtractor, and that computed according to definition!\n\nInput node:\n\n" +
								  stringifyTensor(nodeValue));
	}

	static void testMatmulDerivativeExtracting(const mlCore::Tensor& leftTensor,
											   const mlCore::Tensor& rightTensor,
											   const mlCore::Tensor& outer,
											   const std::pair<mlCore::Tensor, mlCore::Tensor>& expected)
	{
		using mlCore::autoDiff::binaryOperators::BinaryOperator;

		const auto leftNode = std::make_shared<mlCore::autoDiff::Constant>(leftTensor);
		const auto rightNode = std::make_shared<mlCore::autoDiff::Constant>(rightTensor);

		const auto operationResult = std::dynamic_pointer_cast<mlCore::autoDiff::binaryOperators::BinaryOperator>(
			mlCore::autoDiff::binaryOperations::matmul(leftNode, rightNode));

		if(!operationResult)
		{
			FAIL() << "Matmul operation yielded something other than BinaryOperator!";
		}

		const auto [leftDerivative, rightDerivative] = operationResult->computeDerivative(outer);

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
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace mlCore::autoDiff;

	const UnaryParams params{.tensorShape = {3, 3}, .initializer = std::make_unique<RangeTensorInitializer<double>>(-3.0, .7)};

	testUnaryOperationDerivative([](NodePtr node) { return nodesActivations::relu(node); }, params);
}

TEST_F(TestDerivativeExtractor, testLnDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace mlCore::autoDiff;

	const UnaryParams params{.tensorShape = {3, 3}, .initializer = std::make_unique<RangeTensorInitializer<double>>(0.1, .7)};

	testUnaryOperationDerivative([](NodePtr node) { return unaryOperations::ln(node); }, params);
}

TEST_F(TestDerivativeExtractor, testSigmoidDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace mlCore::autoDiff;

	const UnaryParams params{.tensorShape = {5, 5}, .initializer = std::make_unique<RangeTensorInitializer<double>>(-9.0, .7)};

	testUnaryOperationDerivative([](NodePtr node) { return nodesActivations::sigmoid(node); }, params);
}

TEST_F(TestDerivativeExtractor, testMultiplyDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace mlCore::autoDiff;

	const BinaryParams params{.leftTensorShape = {3, 5},
							  .leftInitializer = std::make_unique<RangeTensorInitializer<double>>(-4, .8),
							  .rightTensorShape = {3, 5},
							  .rightInitializer = std::make_unique<RangeTensorInitializer<double>>(5, .3)};

	testBinaryOperationDerivative([](NodePtr left, NodePtr right) { return binaryOperations::multiply(left, right); }, params);
}

TEST_F(TestDerivativeExtractor, testDivideDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace mlCore::autoDiff;

	const BinaryParams params{.leftTensorShape = {3, 5},
							  .leftInitializer = std::make_unique<RangeTensorInitializer<double>>(-4, .7),
							  .rightTensorShape = {3, 5},
							  .rightInitializer = std::make_unique<RangeTensorInitializer<double>>(5, .3)};

	testBinaryOperationDerivative([](NodePtr left, NodePtr right) { return binaryOperations::divide(left, right); }, params);
}

TEST_F(TestDerivativeExtractor, testAddDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace mlCore::autoDiff;

	const BinaryParams params{.leftTensorShape = {3, 5},
							  .leftInitializer = std::make_unique<RangeTensorInitializer<double>>(-4, .7),
							  .rightTensorShape = {3, 5},
							  .rightInitializer = std::make_unique<RangeTensorInitializer<double>>(5, .3)};

	testBinaryOperationDerivative([](NodePtr left, NodePtr right) { return binaryOperations::add(left, right); }, params);
}

TEST_F(TestDerivativeExtractor, testSubtractDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace mlCore::autoDiff;

	const BinaryParams params{.leftTensorShape = {3, 5},
							  .leftInitializer = std::make_unique<RangeTensorInitializer<double>>(-4, .7),
							  .rightTensorShape = {3, 5},
							  .rightInitializer = std::make_unique<RangeTensorInitializer<double>>(5, .3)};

	testBinaryOperationDerivative([](NodePtr left, NodePtr right) { return binaryOperations::subtract(left, right); }, params);
}

TEST_F(TestDerivativeExtractor, testPowerDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace mlCore::autoDiff;

	const auto powerLambda = [](NodePtr left, NodePtr right) { return binaryOperations::power(left, right); };

	const BinaryParams paramsWithLeftScalar{.leftTensorShape = {3, 5},
											.leftInitializer = std::make_unique<RangeTensorInitializer<double>>(1.3, .1),
											.rightTensorShape = {3, 5},
											.rightInitializer = std::make_unique<RangeTensorInitializer<double>>(5, .1)};

	testBinaryOperationDerivative(powerLambda, paramsWithLeftScalar);
}

TEST_F(TestDerivativeExtractor, testMatmulDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	mlCore::Tensor leftInput({4, 3});
	leftInput.fill(RangeTensorInitializer<double>(-4, .7));

	mlCore::Tensor rightInput({3, 2});
	rightInput.fill(RangeTensorInitializer<double>(5, .3));

	mlCore::Tensor outerDerivative({4, 2});
	outerDerivative.fill({1, 2, 3, 4}, true);

	mlCore::Tensor leftExpected({4, 3});
	leftExpected.fill({15.6, 17.4, 19.2, 36.2, 40.4, 44.6, 15.6, 17.4, 19.2, 36.2, 40.4, 44.6});

	mlCore::Tensor rightExpected({3, 2});
	rightExpected.fill({-2.6, -6, 3, 2.4, 8.6, 10.8});

	testMatmulDerivativeExtracting(leftInput, rightInput, outerDerivative, std::pair{leftExpected, rightExpected});
}

} // namespace
