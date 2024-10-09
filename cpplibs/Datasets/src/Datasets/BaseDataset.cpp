#include "Datasets/BaseDataset.h"

#include <random>

#include <LoggingLib/LoggingLib.hpp>

namespace datasets
{
namespace
{
std::vector<size_t> generateIndices(size_t size)
{
	std::vector<size_t> indices(size);
	std::iota(indices.begin(), indices.end(), 0);
	return indices;
}
} // namespace

BaseDataset::BaseDataset(batchProviders::IBatchProviderPtr batchProvider, size_t batchSize, bool shuffle)
	: _batchProvider(std::move(batchProvider))
	, _samplesIndices(generateIndices(_batchProvider->getNumberOfSamples()))
	, _batchSize(batchSize)
	, _shuffle(shuffle)
	, _currentBatchIndex(0)
{
	_resetState();
}

std::vector<mlCore::Tensor> BaseDataset::getNextBatch()
{
	if(!hasNextBatch())
	{
		LOG_ERROR("BaseDataset", "No more batches available!");
	}

	const auto firstIndex = _currentBatchIndex * _batchSize;
	const auto lastIndex = firstIndex + _batchSize;
	std::vector<size_t> indices(_samplesIndices.cbegin() + firstIndex, _samplesIndices.cbegin() + lastIndex);

	++_currentBatchIndex;

	if(_batchSize > 0)
	{
		return _batchProvider->getBatch(indices);
	}

	auto batch = _batchProvider->getBatch(indices);
	const auto batchSpec = _batchProvider->getBatchSpecification();

	for(size_t i = 0; i < batch.size(); ++i)
	{
		batch[i].reshape(batchSpec[i]);
	}

	return batch;
}

void BaseDataset::_resetState()
{
	_currentBatchIndex = 0;

	if(_shuffle)
	{
		std::shuffle(_samplesIndices.begin(), _samplesIndices.end(), std::mt19937(std::random_device()()));
	}
}
} // namespace datasets