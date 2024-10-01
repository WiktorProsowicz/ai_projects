#include "BatchProviders/SerializedTensorsProvider.h"

#include <MLCore/TensorOperations.h>

namespace datasets::batchProviders
{
SerializedTensorsProvider::SerializedTensorsProvider(const std::vector<std::string>& paths,
													 const bool storeInRam)
{
	_serializers.reserve(paths.size());
	std::transform(paths.begin(),
				   paths.end(),
				   std::back_inserter(_serializers),
				   [](const std::string& path) { return mlCore::io::TensorsSerializer::open(path); });

	_validateSerializers();

	if(storeInRam)
	{
		_fillCache();
	}

	std::transform(_serializers.begin(),
				   _serializers.end(),
				   std::back_inserter(_batchSpec),
				   [](const auto& serializer) { return serializer->getTensorHandles()[0]->getShape(); });
}

std::vector<mlCore::Tensor> SerializedTensorsProvider::getBatch(const std::vector<size_t>& samplesIndices)
{
	std::vector<mlCore::Tensor> batch;
	batch.reserve(samplesIndices.size() * _serializers.size());

	for(size_t i = 0; i < _serializers.size(); ++i)
	{
		const auto tensors = _retrieveTensors(i, samplesIndices);
		batch.emplace_back(mlCore::TensorOperations::stack(tensors, 0));
	}

	return batch;
}

std::vector<mlCore::Tensor>
SerializedTensorsProvider::_retrieveTensors(size_t serializerIdx, const std::vector<size_t>& samplesIndices)
{
	std::vector<mlCore::Tensor> tensors;
	tensors.reserve(samplesIndices.size());

	if(_cache.has_value())
	{
		const auto& cache = _cache.value()[serializerIdx];
		std::transform(samplesIndices.begin(),
					   samplesIndices.end(),
					   std::back_inserter(tensors),
					   [&cache](size_t idx) { return cache.tensors[idx]; });
	}
	else
	{
		const auto handles = _serializers[serializerIdx]->getTensorHandles();
		std::transform(samplesIndices.begin(),
					   samplesIndices.end(),
					   std::back_inserter(tensors),
					   [&handles](size_t idx) { return handles[idx]->get(); });
	}

	return tensors;
}

void SerializedTensorsProvider::_validateSerializers() const
{
	const auto nTensorsInFirst = _serializers[0]->getTensorHandles().size();

	if(std::any_of(std::next(_serializers.begin()),
				   _serializers.end(),
				   [nTensorsInFirst](const auto& serializer)
				   { return serializer->getTensorHandles().size() != nTensorsInFirst; }))
	{
		LOG_ERROR("Datasets::SerializedTensorsProvider",
				  "Tensor databases contain different number of tensors!");
	}

	for(const auto& serializer : _serializers)
	{
		const auto handles = serializer->getTensorHandles();

		if(handles.empty())
		{
			LOG_ERROR("Datasets::SerializedTensorsProvider", "Provided an empty input tensors database!");
		}

		const auto firstShape = handles[0]->getShape();

		if(std::any_of(std::next(handles.begin()),
					   handles.end(),
					   [&firstShape](const auto& handle) { return handle->getShape() != firstShape; }))
		{
			LOG_ERROR("Datasets::SerializedTensorsProvider",
					  "Tensor database contains tensors of different shapes!");
		}
	}
}

void SerializedTensorsProvider::_fillCache()
{
	_cache = std::vector<detail::TensorsCache>(_serializers.size());

	for(size_t i = 0; i < _serializers.size(); ++i)
	{
		const auto handles = _serializers[i]->getTensorHandles();
		auto& cache = _cache.value()[i];

		cache.tensors.reserve(handles.size());
		std::transform(handles.begin(),
					   handles.end(),
					   std::back_inserter(cache.tensors),
					   [&serializer = _serializers[i]](const auto& handle) { return handle->get(); });
	}
}
} // namespace datasets::batchProviders