#include "BatchProviders/BatchProviders.h"

#include "BatchProviders/SerializedTensorsProvider.h"

namespace datasets::batchProviders
{
IBatchProviderPtr getFromSerializedPaths(const std::vector<std::string>& paths, const bool storeInRam)
{
	if(paths.empty())
	{
		LOG_ERROR("Datasets::BatchProviders", "Cannot create a batch provider from an empty list of paths!");
	}

	return std::make_shared<SerializedTensorsProvider>(paths, storeInRam);
}
} // namespace datasets::batchProviders