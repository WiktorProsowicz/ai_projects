/**********************
 * Test suite for 'ai_projects'
 * 
 * Copyright (c) 2023
 * 
 * by Wiktor Prosowicz
 **********************/

// __Tested headers__
#include <MLCore/BasicTensorSlice.h>

// __External software headers__
#include <gtest/gtest.h>

// __Own software headers__
#include <MLCore/BasicTensor.h>
#include <MLCore/TensorOperations.h>

namespace
{
/*****************************
 * 
 * Test Fixture TestBasicTensorSlice
 * 
 *****************************/

class TestBasicTensorSlice : public ::testing::Test
{
protected:
	/// Makes tensor based on provided indices and checks if it is serialized correctly.
	static ::testing::AssertionResult correctlyPrints(const std::vector<std::pair<size_t, size_t>>& indices,
													  const std::string& expectedSerialization)
	{
		auto tensor = spawnTestTensor_3_2_3();
		const auto slice = tensor.slice(indices);

		const auto serialized = serializeSlice(slice);

		if(serialized == expectedSerialization)
		{
			return ::testing::AssertionSuccess();
		}
		else
		{
			return ::testing::AssertionFailure() << "Expected serialization: " << expectedSerialization << "\n"
												 << "Actual serialization: " << serialized;
		}
	}

	/// Moves the given `startIter` `expectedValues`.size() times with the given `offset` and compares the consecutive `expectedValues` with the values referenced by the iterator.
	static ::testing::AssertionResult incrementingIteratorProducesCorrectValues(mlCore::SlicedTensorIterator<double>& startIter,
																				const size_t offset,
																				const std::vector<double>& expectedValues)
	{
		for(const auto& expectedValue : expectedValues)
		{
			if(std::abs(expectedValue - *startIter) > 1e-6)
			{
				return ::testing::AssertionFailure() << fmt::format("Expected {}, got {}.", expectedValue, *startIter);
			}

			startIter += offset;
		}

		return ::testing::AssertionSuccess();
	}

	/// Creates tensor slice according to `indices` and compares its values with `expectedValues`.
	static ::testing::AssertionResult spansCorrectValues(const mlCore::BasicTensorSlice<double>& slice,
														 const std::vector<double>& expectedValues)
	{
		for(auto [sliceIter, expIter, itemIdx] = std::tuple{slice.begin(), expectedValues.cbegin(), 0};
			sliceIter < slice.end() && expIter < expectedValues.cend();
			++sliceIter, ++expIter, itemIdx++)
		{
			if(std::abs(*sliceIter - *expIter) > 1e-6)
			{
				return ::testing::AssertionFailure() << fmt::format("Expected values ({}) but on position {} found {}.",
																	fmt::join(expectedValues, ", "),
																	itemIdx,
																	*sliceIter);
			}
		}

		return ::testing::AssertionSuccess();
	}

	/// See incrementingIteratorProducesCorrectValues.
	static ::testing::AssertionResult decrementingIteratorProducesCorrectValues(mlCore::SlicedTensorIterator<double>& startIter,
																				const size_t offset,
																				const std::vector<double>& expectedValues)
	{
		for(const auto& expectedValue : expectedValues)
		{
			if(std::abs(expectedValue - *startIter) > 1e-6)
			{
				return ::testing::AssertionFailure() << fmt::format("Expected {}, got {}.", expectedValue, *startIter);
			}

			startIter -= offset;
		}

		return ::testing::AssertionSuccess();
	}

	/// Checks if a given tensor has expected values.
	static ::testing::AssertionResult tensorHasExpectedValues(const mlCore::Tensor& tensor, const std::vector<double>& values)
	{
		if(tensor.size() != values.size())
		{
			return ::testing::AssertionFailure()
				   << fmt::format("Tensor contains {} values but expected {}!", tensor.size(), values.size());
		}

		auto valuesIter = values.begin();
		for(const auto value : tensor)
		{
			if(value != *valuesIter)
			{
				return ::testing::AssertionFailure() << "Tensor value: " << value << ", expected one: " << *valuesIter << "\n";
			}
			valuesIter++;
		}

		return ::testing::AssertionSuccess();
	}

	/// Composes two tensor slices from provided indices and performs assignment operation. Then checks if the spanned tensor's values are the same as expected.
	static ::testing::AssertionResult performsAssignmentProperly(const mlCore::SliceIndices& lhsIndices,
																 const mlCore::BasicTensorSlice<double>& rhsSlice,
																 const std::vector<double>& expectedValues)
	{
		auto lhsTensor = spawnTestTensor_2_2_2();

		auto lhsSlice = lhsTensor.slice(lhsIndices);

		lhsSlice.assign(rhsSlice);

		return tensorHasExpectedValues(lhsTensor, expectedValues);
	}

	static mlCore::Tensor spawnTestTensor_2_2_2()
	{
		// clang-format off

		using Arr = mlCore::TensorArr<double>;

		return mlCore::TensorOperations::makeTensor(
			            Arr{Arr{
                                Arr{1.0, 2.0},
                                Arr{3.0, 4.0}},
                            Arr{
                                Arr{5.0, 6.0},
                                Arr{7.0, 8.0}}});

		// clang-format on
	}

	static mlCore::Tensor spawnTestTensor_3_2_3()
	{
		// clang-format off

		using Arr = mlCore::TensorArr<double>;

		return mlCore::TensorOperations::makeTensor(
			            Arr{Arr{
                                Arr{1.1, 2.2, 3.3},
                                Arr{4.4, 5.5, 6.6}},
                            Arr{
                                Arr{7.7, 8.8, 9.9},
                                Arr{10.1, 11.11, 12.12}},
                            Arr{
                                Arr{13.13, 14.14, 15.15},
                                Arr{16.16, 17.17, 18.18}}});

		// clang-format on
	}

	static mlCore::Tensor spawnTestTensor_3_2_5()
	{
		// clang-format off

		using Arr = mlCore::TensorArr<double>;

		return mlCore::TensorOperations::makeTensor(
			            Arr{Arr{
                                Arr{1.1, 2.2, 3.3, 4.4, 5.5},
                                Arr{6.6, 7.7, 8.8, 9.9, 10.10}},
                            Arr{
                                Arr{11.11, 12.12, 13.13, 14.14, 15.15},
                                Arr{16.16, 17.17, 18.18, 19.19, 20.20}},
                            Arr{
                                Arr{21.21, 22.22, 23.23, 24.24, 25.25},
                                Arr{26.26, 27.27, 28.28, 29.29, 30.30}}});

		// clang-format on
	}

private:
	static std::string serializeSlice(const mlCore::TensorSlice& slice)
	{
		std::ostringstream oss;
		oss << slice;
		return oss.str();
	}
};

/*****************************
 * 
 * Particular test calls
 * 
 *****************************/

TEST_F(TestBasicTensorSlice, ProducesCorrectBeginAndEndIterators)
{
	auto tensor = spawnTestTensor_3_2_3();
	const auto slice = tensor.slice({{1, 2}, {0, 2}, {0, 2}});

	const auto beg = slice.begin();
	const auto end = slice.end();

	ASSERT_EQ(*beg, 7.7);
	ASSERT_EQ(*end, 12.12);
}

TEST_F(TestBasicTensorSlice, IteratorIncrementsProperly)
{
	{
		auto tensor = spawnTestTensor_3_2_5();

		const auto slice = tensor.slice({{1, 2}, {0, 2}, {1, 4}});

		{
			auto sliceIter = slice.begin();

			EXPECT_TRUE(incrementingIteratorProducesCorrectValues(
				sliceIter, 1, {12.12, 13.13, 14.14, 17.17, 18.18, 19.19, 20.20, 21.21, 22.22, 23.23}));
		}

		{
			auto sliceIter = slice.begin();

			EXPECT_TRUE(incrementingIteratorProducesCorrectValues(sliceIter, 3, {12.12, 17.17, 20.20, 23.23, 26.26}));
		}
	}

	{
		auto tensor = spawnTestTensor_3_2_3();

		const auto slice = tensor.slice({{0, 2}, {0, 1}, {0, 3}});

		{
			auto sliceIter = slice.begin();

			EXPECT_TRUE(
				incrementingIteratorProducesCorrectValues(sliceIter, 1, {1.1, 2.2, 3.3, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12}));
		}
	}
}

TEST_F(TestBasicTensorSlice, IteratorDecrementsProperly)
{
	{
		auto tensor = spawnTestTensor_3_2_5();

		const auto slice = tensor.slice({{1, 2}, {0, 2}, {1, 4}});

		{
			auto sliceIter = slice.end();

			EXPECT_TRUE(decrementingIteratorProducesCorrectValues(
				sliceIter, 1, {20.20, 19.19, 18.18, 17.17, 14.14, 13.13, 12.12, 11.11, 10.10, 9.9}));
		}

		{
			auto sliceIter = slice.end();

			EXPECT_TRUE(decrementingIteratorProducesCorrectValues(sliceIter, 3, {20.20, 17.17, 12.12, 9.9, 6.6, 3.3}));
		}
	}

	{
		auto tensor = spawnTestTensor_3_2_3();

		const auto slice = tensor.slice({{1, 3}, {0, 1}, {0, 3}});

		{
			auto sliceIter = slice.end();

			EXPECT_TRUE(decrementingIteratorProducesCorrectValues(
				sliceIter, 1, {16.16, 15.15, 14.14, 13.13, 9.9, 8.8, 7.7, 6.6, 5.5, 4.4}));
		}
	}
}

TEST_F(TestBasicTensorSlice, SpansCorrectValues)
{
	EXPECT_TRUE(
		spansCorrectValues(spawnTestTensor_3_2_5().slice({{0, 3}, {0, 2}, {1, 2}}), {2.2, 7.7, 12.12, 17.17, 22.22, 27.27}));

	EXPECT_TRUE(
		spansCorrectValues(spawnTestTensor_3_2_5().slice({{0, 2}, {1, 2}, {3, 5}}), {9.9, 10.10, 19.19, 20.20, 29.29, 30.30}));
}

TEST_F(TestBasicTensorSlice, IsSerializationCorrect)
{

	EXPECT_TRUE(correctlyPrints({{1, 2}, {0, 2}}, R"(<BasicTensorSlice dtype=d shape=(1, 2, 3)>
[
 [
  [  7.7,   8.8,   9.9]
  [ 10.1, 11.11, 12.12]
 ]
])"));

	EXPECT_TRUE(correctlyPrints({{0, 3}, {0, 2}, {0, 3}}, R"(<BasicTensorSlice dtype=d shape=(3, 2, 3)>
[
 [
  [  1.1,   2.2,   3.3]
  [  4.4,   5.5,   6.6]
 ]
 [
  [  7.7,   8.8,   9.9]
  [ 10.1, 11.11, 12.12]
 ]
 [
  [13.13, 14.14, 15.15]
  [16.16, 17.17, 18.18]
 ]
])"));

	EXPECT_TRUE(correctlyPrints({{0, 1}, {0, 1}, {0, 1}}, R"(<BasicTensorSlice dtype=d shape=(1, 1, 1)>
[
 [
  [1.1]
 ]
])"));

	EXPECT_TRUE(correctlyPrints({{2, 3}, {1, 2}, {0, 3}}, R"(<BasicTensorSlice dtype=d shape=(1, 1, 3)>
[
 [
  [16.16, 17.17, 18.18]
 ]
])"));
}

TEST_F(TestBasicTensorSlice, PerformsAssignOperationWithRange)
{
	{
		auto tensor = spawnTestTensor_2_2_2();
		auto slice = tensor.slice({{0, 2}, {0, 2}, {1, 2}});
		slice.assign(std::initializer_list{10.0, 20.0});

		EXPECT_TRUE(tensorHasExpectedValues(tensor, {1.0, 10.0, 3.0, 20.0, 5.0, 10.0, 7.0, 20.0}));
	}

	{
		auto tensor = spawnTestTensor_2_2_2();
		auto slice = tensor.slice({{0, 2}, {0, 2}, {0, 1}});
		slice.assign(std::vector{10.0});

		EXPECT_TRUE(tensorHasExpectedValues(tensor, {10.0, 2.0, 10.0, 4.0, 10.0, 6.0, 10.0, 8.0}));
	}

	{
		auto tensor = spawnTestTensor_2_2_2();
		auto slice = tensor.slice({{0, 2}, {0, 1}, {0, 2}});
		slice.assign(std::array{10.0, 20.0, 30.0, 40.0});

		EXPECT_TRUE(tensorHasExpectedValues(tensor, {10.0, 20.0, 3.0, 4.0, 30.0, 40.0, 7.0, 8.0}));
	}
}

TEST_F(TestBasicTensorSlice, DetectsAssignmentOfRangeImpossibleToAlign)
{
	auto tensor = spawnTestTensor_2_2_2();
	auto slice = tensor.slice({{0, 1}, {0, 2}, {0, 2}});

	EXPECT_THROW(slice.assign(std::initializer_list{10.0, 20.0, 30.0}), std::invalid_argument);
	EXPECT_THROW(slice.assign(std::initializer_list{10.0, 20.0, 30.0, 40.0, 50.0}), std::invalid_argument);
}

TEST_F(TestBasicTensorSlice, PerformsAssignOperationWithOtherSlice)
{
	{
		auto tensor = spawnTestTensor_2_2_2() * 10.0;
		auto slice = tensor.slice({{1, 2}, {1, 2}, {0, 2}});

		EXPECT_TRUE(performsAssignmentProperly({{0, 2}, {1, 2}, {0, 2}}, slice, {1., 2., 70., 80., 5., 6., 70., 80.}));
	}

	{
		auto tensor = spawnTestTensor_2_2_2() * 10.0;
		auto slice = tensor.slice({{1, 2}, {0, 1}, {0, 1}});

		EXPECT_TRUE(performsAssignmentProperly({{0, 2}, {0, 1}, {0, 2}}, slice, {50., 50., 3., 4., 50., 50., 7., 8.}));
	}
}

TEST_F(TestBasicTensorSlice, DetectAssignmentOfSliceImpossibleToAlign)
{
	{
		auto tensor = spawnTestTensor_2_2_2();
		auto sliceLhs = tensor.slice({{0, 2}, {0, 1}, {0, 2}});
		auto sliceRhs = tensor.slice({{{0, 1}, {0, 2}, {0, 2}}});

		EXPECT_THROW(sliceLhs.assign(sliceRhs), std::invalid_argument);
	}
}

} // namespace