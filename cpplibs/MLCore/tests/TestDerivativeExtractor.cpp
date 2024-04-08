/**********************
 * Test suite for 'ai_projects'
 *
 * Copyright (c) 2023
 *
 * by Wiktor Prosowicz
 **********************/

#include <iostream>

#include <AutoDiff/GraphNodes.hpp>
#include <AutoDiff/Operations.h>
#include <MLCore/TensorInitializers/RangeTensorInitializer.hpp>
#include <gtest/gtest.h>

namespace
{

/// Concept for functions taking any number of shared pointers of types inheriting from Node and returning
/// analogous type
template <typename Operation, typename... NodePtrs>
concept NodeOperation = requires(NodePtrs... inputNodes, Operation oper) {
	// input types are shared pointers
	(... && std::is_same_v<std::shared_ptr<decltype(*inputNodes)>, decltype(inputNodes)>);

	// pointed types are derived from Node
	(... && std::is_base_of_v<autoDiff::Node, decltype(*inputNodes)>);

	// result type is shared ptr
	std::is_same_v<std::shared_ptr<decltype(*oper(inputNodes...))>, decltype(oper(inputNodes...))>;

	// result pointer points to something derived from Node
	std::is_base_of_v<autoDiff::Node, decltype(*(oper(inputNodes...)))>;
};

struct Params
{
	std::vector<std::vector<size_t>> tensorShapes;
	std::vector<std::shared_ptr<mlCore::tensorInitializers::ITensorInitializer<double>>> initializers;
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
	/// Computes derivative of given operation taking one input tensor (it may be a wrapper for a function
	/// with one node movable and other fixed).
	static mlCore::Tensor
	_computeDefinitionDerivative(const std::function<autoDiff::NodePtr(const autoDiff::NodePtr&)>& oper,
								 const mlCore::Tensor& inputTensor)
	{
		constexpr double kEpsilon = 1e-6;

		const auto backShiftedNode = std::make_shared<autoDiff::Constant>(inputTensor - kEpsilon);
		const auto frontShiftedNode = std::make_shared<autoDiff::Constant>(inputTensor + kEpsilon);

		const auto backShiftedResult = oper(backShiftedNode);
		const auto frontShiftedResult = oper(frontShiftedNode);

		return (frontShiftedResult->getValue() - backShiftedResult->getValue()) / (2 * kEpsilon);
	}

	static void _compareTwoDerivatives(const mlCore::Tensor& computedDerivative,
									   const mlCore::Tensor& expDerivative,
									   const std::string& message)
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

	static void
	_testOperationDerivative(std::function<autoDiff::NodePtr(const std::vector<autoDiff::NodePtr>&)> oper,
							 const Params& params)
	{
		std::vector<mlCore::Tensor> inputValues;
		inputValues.reserve(params.tensorShapes.size());

		for(const auto& shape : params.tensorShapes)
		{
			mlCore::Tensor tensor(shape);
			tensor.fill(*params.initializers.front());
			inputValues.push_back(tensor);
		}

		std::vector<autoDiff::NodePtr> inputNodes;
		inputNodes.reserve(inputValues.size());

		for(const auto& tensor : inputValues)
		{
			inputNodes.push_back(std::make_shared<autoDiff::Constant>(tensor));
		}

		auto operationResult = std::dynamic_pointer_cast<autoDiff::Operator>(oper(inputNodes));

		if(!operationResult)
		{
			FAIL() << "Given operation yields something different than OperationPtr!";
		}

		const auto derivatives = operationResult->computeDirectDerivative();

		for(size_t inputIdx = 0; inputIdx < inputNodes.size(); inputIdx++)
		{
			// Operation wrapper reduced to a single variable input.
			const auto lockedOperation = [&oper, &inputNodes, inputIdx](const autoDiff::NodePtr& node)
			{
				std::vector<autoDiff::NodePtr> newInputs = inputNodes;
				newInputs[inputIdx] = node;

				return oper(newInputs);
			};

			// computing derivative according to definition
			const auto defDerivative =
				_computeDefinitionDerivative(lockedOperation, inputNodes[inputIdx]->getValue());

			std::vector<std::string> stringifiedInputs;
			stringifiedInputs.reserve(inputNodes.size());

			std::transform(inputNodes.begin(),
						   inputNodes.end(),
						   std::back_inserter(stringifiedInputs),
						   [](const auto& node) { return stringifyTensor(node->getValue()); });

			_compareTwoDerivatives(derivatives[inputIdx],
								   defDerivative,
								   "Found inequality while comparing DerivativeExtractor result and "
								   "definition derivative with regard to left input!\n\nInputs:\n\n" +
									   fmt::format("{}", fmt::join(stringifiedInputs, "\n\n")));
		}
	}

	static void _testMatmulDerivativeExtracting(const mlCore::Tensor& leftTensor,
												const mlCore::Tensor& rightTensor,
												const mlCore::Tensor& outer,
												const std::pair<mlCore::Tensor, mlCore::Tensor>& expected)
	{
		const auto leftNode = std::make_shared<autoDiff::Constant>(leftTensor);
		const auto rightNode = std::make_shared<autoDiff::Constant>(rightTensor);

		const auto operationResult =
			std::dynamic_pointer_cast<autoDiff::Operator>(autoDiff::ops::matmul(leftNode, rightNode));

		if(!operationResult)
		{
			FAIL() << "Matmul operation yielded something other than BinaryOperator!";
		}

		const auto derivatives = operationResult->computeDerivative(outer);

		ASSERT_EQ(derivatives.size(), 2);

		const auto& [leftExpected, rightExpected] = expected;

		std::stringstream messageGenerator;
		messageGenerator << "This test needs manual check:\n\nInputs:\n\n"
						 << leftTensor << "\n\n"
						 << rightTensor << "\n\nOuter derivative:\n\n"
						 << outer;

		_compareTwoDerivatives(derivatives.front(), leftExpected, messageGenerator.str());

		_compareTwoDerivatives(derivatives.back(), rightExpected, messageGenerator.str());
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
	using namespace autoDiff;

	const Params params{.tensorShapes = {{3, 3}},
						.initializers = {std::make_shared<RangeTensorInitializer<double>>(-3.0, .7)}};

	_testOperationDerivative([](const std::vector<NodePtr>& nodes) { return ops::relu(nodes.front()); },
							 params);
}

TEST_F(TestDerivativeExtractor, testLnDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace autoDiff;

	const Params params{.tensorShapes = {{3, 3}},
						.initializers = {std::make_shared<RangeTensorInitializer<double>>(0.1, .7)}};

	_testOperationDerivative([](const std::vector<NodePtr>& nodes) { return ops::naturalLog(nodes.front()); },
							 params);
}

TEST_F(TestDerivativeExtractor, testSigmoidDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace autoDiff;

	const Params params{.tensorShapes = {{5, 5}},
						.initializers = {std::make_shared<RangeTensorInitializer<double>>(-9.0, .7)}};

	_testOperationDerivative([](const std::vector<NodePtr>& nodes) { return ops::sigmoid(nodes.front()); },
							 params);
}

TEST_F(TestDerivativeExtractor, testMultiplyDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace autoDiff;

	const Params params{.tensorShapes = {{3, 5}, {3, 5}},
						.initializers = {std::make_shared<RangeTensorInitializer<double>>(-4, .8),
										 std::make_shared<RangeTensorInitializer<double>>(5, .3)}};

	_testOperationDerivative(
		[](const std::vector<NodePtr>& nodes) { return ops::multiply(nodes.front(), nodes.back()); }, params);
}

TEST_F(TestDerivativeExtractor, testDivideDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace autoDiff;

	const Params params{.tensorShapes = {{3, 5}, {3, 5}},
						.initializers = {std::make_shared<RangeTensorInitializer<double>>(-4, .7),
										 std::make_shared<RangeTensorInitializer<double>>(5, .3)}};

	_testOperationDerivative(
		[](const std::vector<NodePtr>& nodes) { return ops::divide(nodes.front(), nodes.back()); }, params);
}

TEST_F(TestDerivativeExtractor, testAddDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace autoDiff;

	const Params params{.tensorShapes = {{3, 5}, {3, 5}},
						.initializers = {std::make_shared<RangeTensorInitializer<double>>(-4, .7),
										 std::make_shared<RangeTensorInitializer<double>>(5, .3)}};

	_testOperationDerivative(
		[](const std::vector<NodePtr>& nodes) { return ops::add(nodes.front(), nodes.back()); }, params);
}

TEST_F(TestDerivativeExtractor, testSubtractDerivative)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	using namespace autoDiff;

	const Params params{.tensorShapes = {{3, 5}, {3, 5}},
						.initializers = {std::make_shared<RangeTensorInitializer<double>>(-4, .7),
										 std::make_shared<RangeTensorInitializer<double>>(5, .3)}};

	_testOperationDerivative(
		[](const std::vector<NodePtr>& nodes) { return ops::subtract(nodes.front(), nodes.back()); }, params);
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

	_testMatmulDerivativeExtracting(
		leftInput, rightInput, outerDerivative, std::pair{leftExpected, rightExpected});
}

} // namespace
