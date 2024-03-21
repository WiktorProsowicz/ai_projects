/**********************
 * Test suite for 'ai_projects'
 * 
 * Copyright (c) 2023
 * 
 * by Wiktor Prosowicz
 **********************/

#include <MLCore/TensorOperations.h>
#include <MLCore/TensorInitializers/RangeTensorInitializer.hpp>
#include <MLCore/Utilities.h>
#include <gtest/gtest.h>

/*****************************
 * 
 * Global declarations
 * 
 *****************************/

template <typename OperType>
concept UnaryTensorOperation = requires(OperType oper, const mlCore::Tensor& tensor)
{
	{
		oper(tensor)
		} -> std::same_as<mlCore::Tensor>;
};

struct UnaryTestParams
{

	std::unique_ptr<mlCore::tensorInitializers::ITensorInitializer<double>> initializer;
	std::vector<double> expectedValues;
};

template <typename OperType>
concept BinaryTensorOperation = requires(OperType oper, const mlCore::Tensor& tensor)
{
	{
		oper(tensor, tensor)
		} -> std::same_as<mlCore::Tensor>;
};

struct BinaryTestParams
{
	std::unique_ptr<mlCore::tensorInitializers::ITensorInitializer<double>> leftInitializer;
	std::unique_ptr<mlCore::tensorInitializers::ITensorInitializer<double>> rightInitializer;
	std::vector<double> expectedValues;
};

/*****************************
 * 
 * Test Fixture
 * 
 *****************************/

namespace
{
class TestTensorOperations : public testing::Test
{

protected:
	const std::vector<size_t> testedTensorShape{3, 3, 3};

	static void compareTensors(const mlCore::Tensor& checked, const mlCore::Tensor& expected)
	{
		ASSERT_EQ(checked.shape(), expected.shape());

		for(auto checkedIter = checked.begin(), expectedIter = expected.begin();
			checkedIter < checked.end() && expectedIter < expected.end();
			checkedIter++, expectedIter++)
		{
			ASSERT_NEAR(*checkedIter, *expectedIter, 1e-3)
				<< "\nInequality at position " << std::distance(checkedIter, checked.end()) << " for checked tensor:\n\n"
				<< checked << "\n\nAnd expected one:\n\n"
				<< expected;
		}
	}

	template <UnaryTensorOperation Operation>
	void performUnaryOperationAndCompare(const UnaryTestParams& params, Operation operation) const
	{
		mlCore::Tensor expectedTensor(testedTensorShape);
		expectedTensor.fill(params.expectedValues.begin(), params.expectedValues.end());

		mlCore::Tensor testedTensor(testedTensorShape);
		testedTensor.fill(*params.initializer);
		testedTensor = operation(testedTensor);

		compareTensors(testedTensor, expectedTensor);
	}

	template <BinaryTensorOperation Operation>
	void performBinaryOperationAndCompare(const BinaryTestParams& params, Operation operation) const
	{
		mlCore::Tensor expectedTensor(testedTensorShape);
		expectedTensor.fill(params.expectedValues.begin(), params.expectedValues.end());

		mlCore::Tensor leftTensorInput(testedTensorShape);
		leftTensorInput.fill(*params.leftInitializer);

		mlCore::Tensor rightTensorInput(testedTensorShape);
		rightTensorInput.fill(*params.rightInitializer);

		const auto testedTensor = operation(leftTensorInput, rightTensorInput);

		compareTensors(testedTensor, expectedTensor);
	}
};

/*****************************
 * 
 * Particular test calls
 * 
 *****************************/

TEST_F(TestTensorOperations, testNaturalLogarithm)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	UnaryTestParams params{.initializer = std::make_unique<RangeTensorInitializer<double>>(1.0),
						   .expectedValues = {0.000, 0.693, 1.099, 1.386, 1.609, 1.792, 1.946, 2.079, 2.197,
											  2.303, 2.398, 2.485, 2.565, 2.639, 2.708, 2.773, 2.833, 2.890,
											  2.944, 2.996, 3.045, 3.091, 3.135, 3.178, 3.219, 3.258, 3.296}};

	performUnaryOperationAndCompare(params, mlCore::TensorOperations::ln);
}

TEST_F(TestTensorOperations, testRelu)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;
	UnaryTestParams params{.initializer = std::make_unique<RangeTensorInitializer<double>>(-12.0),
						   .expectedValues = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,	0.0,  0.0,	0.0,  0.0, 1.0,
											  2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0}};

	performUnaryOperationAndCompare(params, mlCore::TensorOperations::relu);
}

TEST_F(TestTensorOperations, testSigmoid)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	UnaryTestParams params{.initializer = std::make_unique<RangeTensorInitializer<double>>(-6.0, .5),
						   .expectedValues = {0.00247, 0.00407, 0.00669, 0.01099, 0.01799, 0.02931, 0.04743, 0.07586, 0.11920,
											  0.18243, 0.26894, 0.37754, 0.50000, 0.62246, 0.73106, 0.81757, 0.88080, 0.92414,
											  0.95257, 0.97069, 0.98201, 0.98901, 0.99331, 0.99593, 0.99753, 0.99850, 0.99909}};

	performUnaryOperationAndCompare(params, mlCore::TensorOperations::sigmoid);
}

TEST_F(TestTensorOperations, testPower)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	BinaryTestParams params{.leftInitializer = std::make_unique<RangeTensorInitializer<double>>(.5, .5),
							.rightInitializer = std::make_unique<RangeTensorInitializer<double>>(-6.0, .5),
							.expectedValues = {64.000,		1.000,		  0.132,	   0.044,	  0.026,	  0.021,
											   0.023,		0.031,		  0.049,	   0.089,	  0.182,	  0.408,
											   1.000,		2.646,		  7.500,	   22.627,	  72.250,	  243.000,
											   857.375,		3162.278,	  12155.062,   48558.704, 201135.719, 861979.333,
											   3814697.266, 17403307.346, 81721509.398}};

	performBinaryOperationAndCompare(params, mlCore::TensorOperations::power);
}

TEST_F(TestTensorOperations, testMakeTensor)
{
	using Arr = mlCore::TensorArr<double>;

	// clang-format off

	const auto createdTensor = mlCore::TensorOperations::makeTensor(
		Arr{
			Arr{
				Arr{0.0, 1.0, 2.0, 3.0},
				Arr{4.0, 5.0, 6.0, 7.0},
				Arr{8.0, 9.0, 10.0, 11.0}},
			Arr{
				Arr{12.0, 13.0, 14.0, 15.0},
				Arr{16.0, 17.0, 18.0, 19.0},
				Arr{20.0, 21.0, 22.0, 23.0}}});

	// clang-format on

	mlCore::Tensor expectedTensor({2, 3, 4});
	expectedTensor.fill(mlCore::tensorInitializers::RangeTensorInitializer<double>(0.0, 1.0));

	compareTensors(createdTensor, expectedTensor);
}

TEST_F(TestTensorOperations, testMakeTensorWithInconsistentShape)
{
	using Arr = mlCore::TensorArr<double>;

	// clang-format off

	EXPECT_THROW(

	const auto createdTensor = mlCore::TensorOperations::makeTensor(
		Arr{
			Arr{
				Arr{0.0, 1.0, 2.0, 3.0},},
			Arr{
				Arr{12.0, 13.0, 14.0, 15.0},
				Arr{16.0, 17.0, 18.0, 19.0},
				Arr{20.0, 21.0}}});

	, std::runtime_error);

	// clang-format on
}

TEST_F(TestTensorOperations, testMakeTensorWithEmptyComponent)
{
	using Arr = mlCore::TensorArr<double>;

	// clang-format off

	EXPECT_THROW(

	const auto createdTensor = mlCore::TensorOperations::makeTensor(
		Arr{
			Arr{
				Arr{},
				Arr{4.0, 5.0, 6.0, 7.0},
				Arr{8.0, 9.0, 10.0, 11.0}},
			Arr{
				Arr{12.0, 13.0, 14.0, 15.0},
				Arr{16.0, 17.0, 18.0, 19.0},
				Arr{20.0, 21.0, 22.0, 23.0}}});

	, std::runtime_error);

	// clang-format on
}

} // namespace
