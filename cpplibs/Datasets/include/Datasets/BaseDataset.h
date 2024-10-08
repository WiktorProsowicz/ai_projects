#ifndef DATASETS_INCLUDE_DATASETS_BASEDATASET_H
#define DATASETS_INCLUDE_DATASETS_BASEDATASET_H

#include <Interfaces/IDataset.hpp>

#include "BatchProviders/IBatchProvider.hpp"

namespace datasets
{
/**
 * @brief Implements basic functionality for a dataset.
 */
class BaseDataset : public interfaces::IDataset
{
public:
	BaseDataset() = delete;

	/**
	 * @brief Constructs a BaseDataset instance.
	 *
	 * @param batchProvider Provides the data in batches when requested.
	 * @param batchSize The number of samples every batch is intended to contain.
	 * @param shuffle Tells whether the dataset should shuffle the samples on every epoch.
	 */
	BaseDataset(batchProviders::IBatchProviderPtr batchProvider,
				size_t batchSize,
				bool shuffle,
				bool trimRemaining);

	BaseDataset(const BaseDataset&) = delete;
	BaseDataset(BaseDataset&&) = delete;
	BaseDataset& operator=(const BaseDataset&) = delete;
	BaseDataset& operator=(BaseDataset&&) = delete;

	~BaseDataset() override = default;

	size_t getBatchSize() const override;

	bool hasNextBatch() const override;

	std::vector<mlCore::Tensor> getNextBatch() override;

	size_t getNumberOfBatches() const override;

	void resetState() override;
};
} // namespace datasets

#endif