#ifndef INTERFACES_INCLUDE_INTERFACES_DATASET_HPP
#define INTERFACES_INCLUDE_INTERFACES_DATASET_HPP

#include <vector>

#include <MLCore/BasicTensor.h>

namespace interfaces
{
/**
 * @brief Interface for classes holding preprocessed data harvested from a particular source.
 *
 * @details IDataset instances are intended to be used as input providers for machine learning models.
 */
class IDataset
{
public:
	IDataset() = default;

	IDataset(const IDataset&) = default;
	IDataset(IDataset&&) = default;
	IDataset& operator=(const IDataset&) = default;
	IDataset& operator=(IDataset&&) = default;

	virtual ~IDataset() = default;

	/**
	 * @brief Returns the number of samples every batch is intended to contain.
	 *
	 * @details By design, a 0 batch size means the dataset provides each time a single sample without the
	 * additional dimension.
	 */
	virtual size_t getBatchSize() const = 0;

	/**
	 * @brief Tells whether the dataset has more batches to provide before being reset.
	 */
	virtual bool hasNextBatch() const = 0;

	/**
	 * @brief Returns the next batch of samples.
	 *
	 * @details The number of returned tensors depends on the concrete dataset's specification.
	 */
	virtual std::vector<mlCore::Tensor> getNextBatch() = 0;

	/**
	 * @brief Returns the number of batches the dataset contains.
	 */
	virtual size_t getNumberOfBatches() const = 0;

	/**
	 * @brief Resets the dataset's internal state and prepares it for a new iteration.
	 */
	virtual void resetState() = 0;
};

/// @brief Shared pointer to an IDataset instance.
using IDatasetPtr = std::shared_ptr<IDataset>;
} // namespace interfaces

#endif