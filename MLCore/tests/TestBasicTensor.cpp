#include <LoggingLib/LoggingLib.h>
#include <MLCore/BasicTensor.h>
#include <gtest/gtest.h>
#include <iostream>

namespace
{

class TestBasicTensor : public testing::Test
{
protected:
	void checkTensorEquality(const mlCore::Tensor& tensor1, const mlCore::Tensor& tensor2)
	{
		ASSERT_EQ(tensor1.nDimensions(), tensor2.nDimensions());
		ASSERT_EQ(tensor1.size(), tensor2.size());
		ASSERT_EQ(tensor1.shape(), tensor2.shape());

		auto iter1 = tensor1.begin();
		auto iter2 = tensor2.begin();
		for(; (iter1 < tensor1.end()) && (iter2 < tensor2.end()); iter1++, iter2++)
		{
			EXPECT_DOUBLE_EQ(*iter1, *iter2);
		}
	}

	void isTensorEmpty(const mlCore::Tensor& tensor)
	{
		ASSERT_EQ(tensor.shape(), std::vector<size_t>{});
		ASSERT_EQ(tensor.size(), 0);
		ASSERT_EQ(tensor.nDimensions(), 0);
	}

	void checkTensorValues(const mlCore::Tensor& tensor,
						   const std::vector<double>& values,
						   const std::string message = "")
	{
		ASSERT_EQ(tensor.size(), values.size());

		auto valuesIter = values.begin();
		for(const auto value : tensor)
		{
			// std::cout << value << " " << *valuesIter << " ";
			// EXPECT_DOUBLE_EQ(value, *valuesIter) << message;
			if(value != *valuesIter)
			{
				ADD_FAILURE() << "Tensor value: " << value << ", expected one: " << *valuesIter
							  << "\n";
			}
			valuesIter++;
		}
	}
};

TEST_F(TestBasicTensor, testConstructorWithShape)
{

	constexpr uint8_t nTestCases = 5;
	const std::vector<std::vector<uint64_t>> shapes{
		{1, 1, 2, 3, 4}, {50, 2, 50}, {3, 5, 3, 1, 3, 5, 6, 7, 5, 3, 2}, {1}, {}};
	const std::vector<uint64_t> sizes{24, 5000, 850500, 1, 1};

	for(uint8_t i = 0; i < nTestCases; i++)
	{
		const mlCore::Tensor tensor(shapes[i]);

		ASSERT_EQ(tensor.nDimensions(), shapes[i].size());
		ASSERT_EQ(tensor.size(), sizes[i]);
		ASSERT_EQ(tensor.shape(), shapes[i]);
	}
}

TEST_F(TestBasicTensor, testConstructorWithInitialValue)
{
	constexpr uint8_t nTestCases = 5;

	const std::vector<std::vector<size_t>> shapes{{1, 2, 3, 4}, {20}, {}, {1}, {3, 3, 10}};
	const std::vector<double> values{4.5, 12.5, 0.54, -20.0, 2000.005};

	for(uint8_t i = 0; i < nTestCases; i++)
	{
		const mlCore::Tensor tensor(shapes[i], values[i]);

		for(const auto value : tensor)
			EXPECT_DOUBLE_EQ(value, values[i]);
	}
}

TEST_F(TestBasicTensor, testFillingTensor)
{
	// properly filled tensor
	mlCore::Tensor tensorProperlyFilled(std::vector<size_t>{1, 2, 3, 4});
	const std::vector<double> values{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
	tensorProperlyFilled.fill(values.begin(), values.end());

	checkTensorValues(tensorProperlyFilled, values);

	// tensor filled with too few values
	mlCore::Tensor tensorUnderfilled(std::vector<size_t>{10});
	EXPECT_THROW(tensorUnderfilled.fill({1, 2, 3, 4, 5}), std::out_of_range);

	// tensor filled with wrapped values
	mlCore::Tensor tensorWithWrappedValues(std::vector<size_t>{2, 3});
	tensorWithWrappedValues.fill({1, 2}, true);

	const std::vector<double> wrappedValues{1, 2, 1, 2, 1, 2};

	checkTensorValues(tensorWithWrappedValues, wrappedValues);
}

TEST_F(TestBasicTensor, testDisplayingTensor)
{
	constexpr uint8_t nTestCases = 2;

	const std::vector<std::vector<size_t>> shapes{{2, 2, 3}, {3, 3, 3, 1}};

	const std::vector<std::vector<double>> tensorsValues{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{.1,  .2,  .3,	.4,	 .5,  .6,  .7,	.8,	 .9,  .10, .11, .12, .13, .14,
		 .15, .16, .17, .18, .19, .20, .21, .22, .23, .24, .25, .26, .27}};

	// clang-format off
	const std::vector<std::string> reprs{
R"(<BasicTensor dtype=d shape=(2, 2, 3, )>
[
 [
  [1, 2, 3]
  [4, 5, 6]
 ]
 [
  [7, 8, 9]
  [10, 11, 12]
 ]
])",

R"(<BasicTensor dtype=d shape=(3, 3, 3, 1, )>
[
 [
  [
   [0.1]
   [0.2]
   [0.3]
  ]
  [
   [0.4]
   [0.5]
   [0.6]
  ]
  [
   [0.7]
   [0.8]
   [0.9]
  ]
 ]
 [
  [
   [0.1]
   [0.11]
   [0.12]
  ]
  [
   [0.13]
   [0.14]
   [0.15]
  ]
  [
   [0.16]
   [0.17]
   [0.18]
  ]
 ]
 [
  [
   [0.19]
   [0.2]
   [0.21]
  ]
  [
   [0.22]
   [0.23]
   [0.24]
  ]
  [
   [0.25]
   [0.26]
   [0.27]
  ]
 ]
])"};
	// clang-format on

	for(uint8_t i = 0; i < nTestCases; i++)
	{
		std::ostringstream displayStream;
		mlCore::Tensor tensor(shapes[i]);

		tensor.fill(tensorsValues[i].begin(), tensorsValues[i].end());
		displayStream << tensor;

		if(strcmp(displayStream.str().c_str(), reprs[i].c_str()) != 0)
		{
			ADD_FAILURE() << "Differences between streamed representation of tensor: \n\n"
						  << displayStream.str().c_str() << "\n\nand expected one:\n\n"
						  << reprs[i].c_str();
		}
	}
}

TEST_F(TestBasicTensor, testCopy)
{
	mlCore::Tensor tensor1(std::vector<size_t>{2, 3, 4});

	tensor1.fill(mlCore::RangeTensorInitializer<double>(0));
	mlCore::Tensor tensor2 = tensor1;

	checkTensorEquality(tensor1, tensor2);

	mlCore::Tensor tensor3(std::vector<size_t>{});
	tensor3 = tensor1;

	checkTensorEquality(tensor1, tensor3);
}

TEST_F(TestBasicTensor, testMove)
{
	mlCore::Tensor tensorBase(std::vector<size_t>{2, 3, 4});

	tensorBase.fill(mlCore::RangeTensorInitializer<double>(0));
	mlCore::Tensor tensorBase_1 = tensorBase, tensorBase_2 = tensorBase;

	mlCore::Tensor tensorMovedByConstructor = std::move(tensorBase_1);

	checkTensorEquality(tensorMovedByConstructor, tensorBase);
	isTensorEmpty(tensorBase_1);

	mlCore::Tensor tensorMovedByAssignment(std::vector<size_t>{});
	tensorMovedByAssignment = std::move(tensorBase_2);

	checkTensorEquality(tensorMovedByAssignment, tensorBase);
	isTensorEmpty(tensorBase_2);
}

TEST_F(TestBasicTensor, testAssignFunction)
{

	const std::vector<std::vector<size_t>> shapes{{2, 3}, {5, 2}};
	const std::vector<std::vector<double>> initialValues{{1, 1, 1, 1, 1, 1},
														 {10, 9, 8, 7, 6, 5, 4, 3, 2, 1}};

	const std::vector<std::vector<double>> endValues{{10, 11, 12, 13, 14, 15},
													 {10, 9, 8, 7, 0, 5, 0, 3, 2, 1}};

	mlCore::Tensor tensor1(shapes[0]);
	tensor1.fill(initialValues[0].begin(), initialValues[0].end());

	tensor1.assign({{0, 2}, {0, 3}}, {10, 11, 12, 13, 14, 15});

	checkTensorValues(tensor1, endValues[0]);

	mlCore::Tensor tensor2(shapes[1]);
	tensor2.fill(initialValues[1].begin(), initialValues[1].end());

	tensor2.assign({{2, 4}, {0, 1}}, {0, 0});

	checkTensorValues(tensor2, endValues[1]);
}

TEST_F(TestBasicTensor, testOperatorsWithoutBroadcasting)
{

	const std::vector<size_t> shape{2, 5};

	const std::vector<double> values1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	const std::vector<double> values2{3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	const std::vector<double> values3{1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6};
	const std::vector<double> values4{-0.5, 0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4};
	const std::vector<double> values5{-1, 0, 1, 2, 3, 4, 5, 6, 7, 8};

	mlCore::Tensor modifiedTensor(shape);
	modifiedTensor.fill(values1.begin(), values1.end());
	const mlCore::Tensor factorTensor(shape, 2);

	modifiedTensor = modifiedTensor + factorTensor;
	checkTensorValues(modifiedTensor, values2);

	modifiedTensor = modifiedTensor / factorTensor;
	checkTensorValues(modifiedTensor, values3);

	modifiedTensor = modifiedTensor - factorTensor;
	checkTensorValues(modifiedTensor, values4);

	modifiedTensor = modifiedTensor * factorTensor;
	checkTensorValues(modifiedTensor, values5);
}

TEST_F(TestBasicTensor, testOperatorsWithBroadcasting)
{
	const std::vector<size_t> shape1{2, 1, 7};
	const std::vector<size_t> shape2{3, 1};

	const std::vector<size_t> resShape{2, 3, 7};

	mlCore::Tensor tensor1(shape1);
	tensor1.fill({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});

	mlCore::Tensor tensor2(shape2);
	tensor2.fill({1, 2, 3});

	mlCore::Tensor resTensor = tensor1 + tensor2;

	ASSERT_EQ(resTensor.shape(), resShape);
	const std::vector<double> resValues{2,	3,	4,	5,	6,	7,	8,	3,	4,	5,	6,	7,	8,	9,
										4,	5,	6,	7,	8,	9,	10, 9,	10, 11, 12, 13, 14, 15,
										10, 11, 12, 13, 14, 15, 16, 11, 12, 13, 14, 15, 16, 17};

	checkTensorValues(resTensor, resValues);
}

TEST_F(TestBasicTensor, testMatrixMultiplicationClassicMatrices)
{
	mlCore::Tensor firstTensor(std::vector<size_t>{3, 2});
	mlCore::Tensor secondTensor(std::vector<size_t>{2, 4});

	firstTensor.fill(mlCore::RangeTensorInitializer<double>(1));
	secondTensor.fill(mlCore::RangeTensorInitializer<double>(1));

	mlCore::Tensor resultTensor = firstTensor.matmul(secondTensor);

	const std::vector<double> expectedValues{11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68};

	checkTensorValues(resultTensor, expectedValues);
}

TEST_F(TestBasicTensor, testMatrixMultiplicationExtended)
{
	mlCore::Tensor firstTensor({2, 4, 2});
	mlCore::Tensor secondTensor({2, 5});

	firstTensor.fill(mlCore::RangeTensorInitializer<double>(1));
	secondTensor.fill(mlCore::RangeTensorInitializer<double>(1));

	auto result = firstTensor.matmul(secondTensor);

	const std::vector<double> expectedValues{13, 16,  19,  22,	25,	 27,  34,  41,	48,	 55,
											 41, 52,  63,  74,	85,	 55,  70,  85,	100, 115,
											 69, 88,  107, 126, 145, 83,  106, 129, 152, 175,
											 97, 124, 151, 178, 205, 111, 142, 173, 204, 235};

	checkTensorValues(result, expectedValues);
}

} // namespace