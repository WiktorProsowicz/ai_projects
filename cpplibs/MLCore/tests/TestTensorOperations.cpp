/**********************
 * Test suite for 'ai_projects'
 *
 * Copyright (c) 2023
 *
 * by Wiktor Prosowicz
 **********************/

#include <MLCore/TensorInitializers/RangeTensorInitializer.hpp>
#include <MLCore/TensorOperations.h>
#include <MLCore/Utilities.h>
#include <gtest/gtest.h>

/*****************************
 *
 * Global declarations
 *
 *****************************/

template <typename OperType>
concept UnaryTensorOperation = requires(OperType oper, const mlCore::Tensor& tensor) {
	{
		oper(tensor)
	} -> std::same_as<mlCore::Tensor>;
};

struct UnaryTestParams
{
	mlCore::Tensor tensor;
	mlCore::Tensor expectedOutput;
};

template <typename OperType>
concept BinaryTensorOperation = requires(OperType oper, const mlCore::Tensor& tensor) {
	{
		oper(tensor, tensor)
	} -> std::same_as<mlCore::Tensor>;
};

struct BinaryTestParams
{
	mlCore::Tensor lhsTensor;
	mlCore::Tensor rhsTensor;
	mlCore::Tensor expectedOutput;
};

mlCore::Tensor initTensor(const std::vector<size_t>& shape,
						  std::unique_ptr<mlCore::tensorInitializers::ITensorInitializer<double>> initializer)
{
	mlCore::Tensor tensor(shape);
	tensor.fill(*initializer);
	return tensor;
}

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
	static void _compareTensors(const mlCore::Tensor& checked, const mlCore::Tensor& expected)
	{
		ASSERT_EQ(checked.shape(), expected.shape());

		for(auto checkedIter = checked.begin(), expectedIter = expected.begin();
			checkedIter < checked.end() && expectedIter < expected.end();
			checkedIter++, expectedIter++)
		{
			ASSERT_NEAR(*checkedIter, *expectedIter, 1e-3)
				<< "\nInequality at position " << std::distance(checkedIter, checked.end())
				<< " for checked tensor:\n\n"
				<< checked << "\n\nAnd expected one:\n\n"
				<< expected;
		}
	}

	template <UnaryTensorOperation Operation>
	void _performUnaryOperationAndCompare(const UnaryTestParams& params, Operation operation) const
	{
		const auto output = operation(params.tensor);

		_compareTensors(output, params.expectedOutput);
	}

	template <BinaryTensorOperation Operation>
	void _performBinaryOperationAndCompare(const BinaryTestParams& params, Operation operation) const
	{
		const auto output = operation(params.lhsTensor, params.rhsTensor);

		_compareTensors(output, params.expectedOutput);
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

	const UnaryTestParams params{
		.tensor = initTensor({3, 3, 3}, std::make_unique<RangeTensorInitializer<double>>(1.0)),
		.expectedOutput = {{3, 3, 3}, {0.000, 0.693, 1.099, 1.386, 1.609, 1.792, 1.946, 2.079, 2.197,
									   2.303, 2.398, 2.485, 2.565, 2.639, 2.708, 2.773, 2.833, 2.890,
									   2.944, 2.996, 3.045, 3.091, 3.135, 3.178, 3.219, 3.258, 3.296}}};

	_performUnaryOperationAndCompare(params, mlCore::TensorOperations::ln);
}

TEST_F(TestTensorOperations, testRelu)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	const UnaryTestParams params{
		.tensor = initTensor({3, 3, 3}, std::make_unique<RangeTensorInitializer<double>>(-12.0)),
		.expectedOutput = {{3, 3, 3},
						   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,	0.0,  0.0,	0.0, 1.0,
							2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0}}};

	_performUnaryOperationAndCompare(params, mlCore::TensorOperations::relu);
}

TEST_F(TestTensorOperations, testSigmoid)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	const UnaryTestParams params{
		.tensor = initTensor({3, 3, 3}, std::make_unique<RangeTensorInitializer<double>>(-6.0, .5)),
		.expectedOutput = {{3, 3, 3}, {0.00247, 0.00407, 0.00669, 0.01099, 0.01799, 0.02931, 0.04743,
									   0.07586, 0.11920, 0.18243, 0.26894, 0.37754, 0.50000, 0.62246,
									   0.73106, 0.81757, 0.88080, 0.92414, 0.95257, 0.97069, 0.98201,
									   0.98901, 0.99331, 0.99593, 0.99753, 0.99850, 0.99909}}};

	_performUnaryOperationAndCompare(params, mlCore::TensorOperations::sigmoid);
}

TEST_F(TestTensorOperations, CorrectlyTransposesMatrix)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	const UnaryTestParams params{
		// 0  1  2  3  4
		// 5  6  7  8  9
		// 10 11 12 13 14
		// 15 16 17 18 19
		.tensor = initTensor({4, 5}, std::make_unique<RangeTensorInitializer<double>>(0.0)),
		// 0  5  10  15
		// 1  6  11  16
		// 2  7  12  17
		// 3  8  13  18
		// 4  9  14  19
		.expectedOutput = {{5, 4}, {0, 5, 10, 15, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14, 19}}};

	_performUnaryOperationAndCompare(
		params, [](const auto& tensor) { return mlCore::TensorOperations::transpose(tensor); });
}

TEST_F(TestTensorOperations, CorrectlyTransposesColumnVector)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	const UnaryTestParams params{
		// 0
		// 5
		// 10
		// 15
		.tensor = initTensor({4}, std::make_unique<RangeTensorInitializer<double>>(0.0, 5.0)),
		// 0  5  10  15
		.expectedOutput = {{1, 4}, {0, 5, 10, 15}}};

	_performUnaryOperationAndCompare(
		params,
		[](const auto& tensor)
		{ return mlCore::TensorOperations::transpose(tensor, mlCore::MatrixSpec::ColumnVector); });
}

TEST_F(TestTensorOperations, CorrectlyTransposesRowVector)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	const UnaryTestParams params{
		// 0 5 10, 15
		.tensor = initTensor({4}, std::make_unique<RangeTensorInitializer<double>>(0.0, 5.0)),
		// 0
		// 5
		// 10
		// 15
		.expectedOutput = {{4, 1}, {0, 5, 10, 15}}};

	_performUnaryOperationAndCompare(
		params,
		[](const auto& tensor)
		{ return mlCore::TensorOperations::transpose(tensor, mlCore::MatrixSpec::RowVector); });
}

TEST_F(TestTensorOperations, testPower)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	const BinaryTestParams params{
		.lhsTensor = initTensor({3, 3, 3}, std::make_unique<RangeTensorInitializer<double>>(.5, .5)),
		.rhsTensor = initTensor({3, 3, 3}, std::make_unique<RangeTensorInitializer<double>>(-6.0, .5)),
		.expectedOutput = {
			{3, 3, 3}, {64.000,	   1.000,	   0.132,	   0.044,		0.026,		  0.021,	   0.023,
						0.031,	   0.049,	   0.089,	   0.182,		0.408,		  1.000,	   2.646,
						7.500,	   22.627,	   72.250,	   243.000,		857.375,	  3162.278,	   12155.062,
						48558.704, 201135.719, 861979.333, 3814697.266, 17403307.346, 81721509.398}}};

	_performBinaryOperationAndCompare(params, mlCore::TensorOperations::power);
}

TEST_F(TestTensorOperations, testMatrixMultiplicationClassicMatrices)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	const BinaryTestParams params{
		.lhsTensor = initTensor({3, 2}, std::make_unique<RangeTensorInitializer<double>>(1.0)),
		.rhsTensor = initTensor({2, 4}, std::make_unique<RangeTensorInitializer<double>>(1.0)),
		.expectedOutput = {{3, 4}, {11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68}}};

	_performBinaryOperationAndCompare(
		params, [](const auto& lhs, const auto& rhs) { return mlCore::TensorOperations::matmul(lhs, rhs); });
}

TEST_F(TestTensorOperations, testMatrixMultiplicationExtended)
{

	using mlCore::tensorInitializers::RangeTensorInitializer;

	const BinaryTestParams params{
		.lhsTensor = initTensor({2, 4, 2}, std::make_unique<RangeTensorInitializer<double>>(1.0)),
		.rhsTensor = initTensor({2, 5}, std::make_unique<RangeTensorInitializer<double>>(1.0)),
		.expectedOutput = {{2, 4, 5}, {13,	16,	 19, 22,  25,  27,	34,	 41,  48,  55,	41,	 52, 63,  74,
									   85,	55,	 70, 85,  100, 115, 69,	 88,  107, 126, 145, 83, 106, 129,
									   152, 175, 97, 124, 151, 178, 205, 111, 142, 173, 204, 235}}};

	_performBinaryOperationAndCompare(
		params, [](const auto& lhs, const auto& rhs) { return mlCore::TensorOperations::matmul(lhs, rhs); });
}

TEST_F(TestTensorOperations, testMatrixMultiplicationWithColumnVector)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	const BinaryTestParams params{
		.lhsTensor = initTensor({2, 5}, std::make_unique<RangeTensorInitializer<double>>(0.0)),
		.rhsTensor = initTensor({5}, std::make_unique<RangeTensorInitializer<double>>(0.0)),
		.expectedOutput = {{2, 1}, {30, 80}}};

	_performBinaryOperationAndCompare(
		params,
		[](const auto& lhs, const auto& rhs)
		{
			return mlCore::TensorOperations::matmul(
				lhs, rhs, mlCore::MatrixSpec::Default, mlCore::MatrixSpec::ColumnVector);
		});
}

TEST_F(TestTensorOperations, testMatrixMultiplicationWithRowVector)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	const BinaryTestParams params{
		.lhsTensor = initTensor({5}, std::make_unique<RangeTensorInitializer<double>>(0.0)),
		.rhsTensor = initTensor({5, 2}, std::make_unique<RangeTensorInitializer<double>>(0.0)),
		.expectedOutput = {{1, 2}, {60, 70}}};

	_performBinaryOperationAndCompare(
		params,
		[](const auto& lhs, const auto& rhs)
		{
			return mlCore::TensorOperations::matmul(
				lhs, rhs, mlCore::MatrixSpec::RowVector, mlCore::MatrixSpec::Default);
		});
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

	_compareTensors(createdTensor, expectedTensor);
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

TEST_F(TestTensorOperations, CorrectlyReduceAddsTensor)
{
	using mlCore::tensorInitializers::RangeTensorInitializer;

	const UnaryTestParams params{.tensor = {{3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11}},
								 .expectedOutput = {{2, 2}, {15, 18, 21, 23}}};

	_performUnaryOperationAndCompare(params,
									 [](const auto& tensor) {
										 return mlCore::TensorOperations::reduceAdd(tensor, {2, 2});
									 });
}
} // namespace
