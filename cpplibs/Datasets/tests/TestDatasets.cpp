#include <Datasets/BaseDataset.h>
#include <gmock/gmock.h>
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

class MockBatchProvider : public datasets::batchProviders::IBatchProvider
{
public:
	MOCK_METHOD(size_t, getNumberOfSamples, (), (const, override));
	MOCK_METHOD(std::vector<mlCore::TensorShape>, getBatchSpecification, (), (const, override));
	MOCK_METHOD(std::vector<mlCore::Tensor>,
				getBatch,
				(const std::vector<size_t>& samplesIndices),
				(override));
};

/**
 * @brief Provides testing functions for classes implementing IBatchProvider.
 */
template <class DatasetClass>
class TestDataset : public ::testing::Test
{
public:
	void setupDataset(const std::shared_ptr<DatasetClass>& dataset)
	{
		_dataset = dataset;
	}

	void assertHasNextBatch() const
	{
		ASSERT_TRUE(_dataset->hasNextBatch());
	}

	void assertHasNoNextBatch() const
	{
		ASSERT_FALSE(_dataset->hasNextBatch());
	}

	void assertNextBatchIsCorrect(const std::vector<mlCore::Tensor>& expectedBatch) const
	{
		const auto batch = _dataset->getNextBatch();

		ASSERT_EQ(batch.size(), expectedBatch.size());

		for(size_t i = 0; i < batch.size(); ++i)
		{
			ASSERT_TRUE(areTensorsEqual(batch[i], expectedBatch[i]));
		}
	}

	void assertNumberOfBatchesIsCorrect(const size_t expectedNumberOfBatches) const
	{
		ASSERT_EQ(_dataset->getNumberOfBatches(), expectedNumberOfBatches);
	}

	void resetState() const
	{
		_dataset->resetState();
	}

private:
	std::shared_ptr<DatasetClass> _dataset;
};

using TestBaseDataset = TestDataset<datasets::BaseDataset>;
} // namespace

TEST_F(TestBaseDataset, PerformsCorrectBatchCycle)
{
	const auto batchProvider = std::make_shared<MockBatchProvider>();

	{
		EXPECT_CALL(*batchProvider, getNumberOfSamples()).WillRepeatedly(::testing::Return(4));

		EXPECT_CALL(*batchProvider, getBatchSpecification())
			.WillRepeatedly(::testing::Return(std::vector<mlCore::TensorShape>{{2, 2}}));

		EXPECT_CALL(*batchProvider, getBatch(std::vector<size_t>{0, 1}))
			.WillRepeatedly(::testing::Return(
				std::vector<mlCore::Tensor>{mlCore::Tensor({2, 2, 2}, {0, 0, 0, 0, 1, 1, 1, 1})}));

		EXPECT_CALL(*batchProvider, getBatch(std::vector<size_t>{2, 3}))
			.WillRepeatedly(::testing::Return(
				std::vector<mlCore::Tensor>{mlCore::Tensor({2, 2, 2}, {2, 2, 2, 2, 3, 3, 3, 3})}));
	}

	setupDataset(std::make_shared<datasets::BaseDataset>(batchProvider, 2, false));
	assertNumberOfBatchesIsCorrect(2);

	constexpr size_t epochs = 2;
	for(size_t epoch = 0; epoch < epochs; epoch++)
	{
		assertHasNextBatch();

		assertNextBatchIsCorrect({mlCore::Tensor({2, 2, 2}, {0, 0, 0, 0, 1, 1, 1, 1})});

		assertHasNextBatch();

		assertNextBatchIsCorrect({mlCore::Tensor({2, 2, 2}, {2, 2, 2, 2, 3, 3, 3, 3})});

		assertHasNoNextBatch();

		resetState();
	}
}
