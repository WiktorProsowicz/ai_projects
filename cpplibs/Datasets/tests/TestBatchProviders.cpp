#include <BatchProviders/SerializedTensorsProvider.h>
#include <gtest/gtest.h>

namespace
{
/// @brief Checks if two tensors are equal.
::testing::AssertionResult areTensorsEqual(const mlCore::Tensor& tensor1, const mlCore::Tensor& tensor2)
{
	if(tensor1.shape() != tensor2.shape())
	{
		return ::testing::AssertionFailure() << fmt::format(
				   "Shapes of the tensors are different: {} vs {}", tensor1.shape(), tensor2.shape());
	}

	if(!std::equal(tensor1.begin(), tensor1.end(), tensor2.begin()))
	{
		return ::testing::AssertionFailure() << "Tensors are different:\n"
											 << tensor1 << "\n\nvs\n\n"
											 << tensor2;
	}

	return ::testing::AssertionSuccess();
}

/**
 * @brief Provides testing functions for classes implementing IBatchProvider.
 */
template <class BatchProviderClass>
class TestBatchProvider : public ::testing::Test
{
protected:
	void _setupBatchProvider(const std::shared_ptr<BatchProviderClass>& batchProvider)
	{
		_batchProvider = batchProvider;
	}

	void _assertCorrectBatchSpec(const std::vector<mlCore::TensorShape>& expectedBatchSpec)
	{
		ASSERT_EQ(_batchProvider->getBatchSpecification(), expectedBatchSpec);
	}

	void _assertCorrectNumberOfSamples(const size_t expectedNumberOfSamples)
	{
		ASSERT_EQ(_batchProvider->getNumberOfSamples(), expectedNumberOfSamples);
	}

	void _assertCorrectBatch(const std::vector<mlCore::Tensor>& expectedBatch)
	{
		const auto batch = _batchProvider->getBatch({0, 1});

		ASSERT_EQ(batch.size(), expectedBatch.size());

		for(size_t i = 0; i < batch.size(); ++i)
		{
			ASSERT_TRUE(areTensorsEqual(batch[i], expectedBatch[i]));
		}
	}

private:
	std::shared_ptr<BatchProviderClass> _batchProvider;
};

using TestSerializedTensorsProvider = TestBatchProvider<datasets::batchProviders::SerializedTensorsProvider>;
} // namespace

TEST_F(TestSerializedTensorsProvider, ReadsCorrectlyFromDisk)
{
	const std::vector<std::string> tensorsPaths{
		fmt::format("{}/SerializedTensorsProvider/tensors1", TEST_DATA_DIR),
		fmt::format("{}/SerializedTensorsProvider/tensors2", TEST_DATA_DIR)};

	_setupBatchProvider(
		std::make_shared<datasets::batchProviders::SerializedTensorsProvider>(tensorsPaths, false));

	_assertCorrectNumberOfSamples(2);
	_assertCorrectBatchSpec({{2, 2}, {2, 3}});
	_assertCorrectBatch({mlCore::Tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}),
						 mlCore::Tensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})});
}

TEST_F(TestSerializedTensorsProvider, ReadsCorrectlyFromRAM)
{
	const std::vector<std::string> tensorsPaths{
		fmt::format("{}/SerializedTensorsProvider/tensors1", TEST_DATA_DIR),
		fmt::format("{}/SerializedTensorsProvider/tensors2", TEST_DATA_DIR)};

	_setupBatchProvider(
		std::make_shared<datasets::batchProviders::SerializedTensorsProvider>(tensorsPaths, true));

	_assertCorrectNumberOfSamples(2);
	_assertCorrectBatchSpec({{2, 2}, {2, 3}});
	_assertCorrectBatch({mlCore::Tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}),
						 mlCore::Tensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})});
}