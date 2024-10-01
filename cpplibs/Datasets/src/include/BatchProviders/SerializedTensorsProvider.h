#ifndef DATASETS_SRC_INCLUDE_BATCHPROVIDERS_SERIALIZEDTENSORSPROVIDER_H
#define DATASETS_SRC_INCLUDE_BATCHPROVIDERS_SERIALIZEDTENSORSPROVIDER_H

#include <optional>

#include <MLCore/TensorIO/TensorsSerializer.h>

#include "BatchProviders/IBatchProvider.hpp"

namespace datasets::batchProviders
{
namespace detail
{
/**
 * @brief Stores a vector of tensors.
 */
struct TensorsCache
{
	std::vector<mlCore::Tensor> tensors{};
};
} // namespace detail

/**
 * @brief Batch provider that reads serialized tensors from disk.
 *
 * @details The provider reads serialized tensors from disk and returns them as batches of tensors.
 */
class SerializedTensorsProvider : public IBatchProvider
{
public:
	SerializedTensorsProvider() = delete;

	/**
	 * @brief SerializedTensorsProvider constructor.
	 *
	 * @param paths Paths to files with serialized tensors. The files should follow the interface specified by
	 * the mlCore::io::TensorsSerializer.
	 * @param storeInRam If true, the tensors will be stored in RAM instead of being read from disk every
	 * time.
	 *
	 */
	SerializedTensorsProvider(const std::vector<std::string>& paths, const bool storeInRam);

	size_t getNumberOfSamples() const override
	{
		return _serializers[0]->getTensorHandles().size();
	}

	std::vector<mlCore::TensorShape> getBatchSpecification() const override
	{
		return _batchSpec;
	}

	std::vector<mlCore::Tensor> getBatch(const std::vector<size_t>& samplesIndices) override;

private:
	void _validateSerializers() const;

	void _fillCache();

	std::vector<mlCore::Tensor> _retrieveTensors(size_t serializerIdx,
												 const std::vector<size_t>& samplesIndices);

	std::vector<std::unique_ptr<mlCore::io::TensorsSerializer>> _serializers{};
	std::optional<std::vector<detail::TensorsCache>> _cache{};
	std::vector<mlCore::TensorShape> _batchSpec{};
};
} // namespace datasets::batchProviders

#endif