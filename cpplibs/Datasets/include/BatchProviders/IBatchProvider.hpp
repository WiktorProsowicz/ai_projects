#ifndef DATASETS_INCLUDE_DATASETS_IBATCHPROVIDER_HPP
#define DATASETS_INCLUDE_DATASETS_IBATCHPROVIDER_HPP

#include <vector>

#include <MLCore/BasicTensor.h>

namespace datasets::batchProviders
{
/**
 * @brief Interface for classes providing batches of data.
 *
 * @details Concrete batch providers shall operate over a particular data source and provide batches of
 * tensors when requested.
 */
class IBatchProvider
{
public:
	IBatchProvider() = default;

	IBatchProvider(const IBatchProvider&) = default;
	IBatchProvider(IBatchProvider&&) = default;
	IBatchProvider& operator=(const IBatchProvider&) = default;
	IBatchProvider& operator=(IBatchProvider&&) = default;

	virtual ~IBatchProvider() = default;

	/**
	 * @brief Returns the total number of samples the provider contains.
	 */
	virtual size_t getNumberOfSamples() const = 0;

	/**
	 * @brief Returns the shapes of the tensors that the batch provider will return.
	 *
	 * @details The returned shapes don't include the batch dimension.
	 */
	virtual std::vector<mlCore::TensorShape> getBatchSpecification() const = 0;

	/**
	 * @brief Compiles a batch of samples specified by their indices.
	 */
	virtual std::vector<mlCore::Tensor> getBatch(const std::vector<size_t>& samplesIndices) = 0;
};

/// @brief Shared pointer to an IBatchProvider instance.
using IBatchProviderPtr = std::shared_ptr<IBatchProvider>;
} // namespace datasets::batchProviders

#endif