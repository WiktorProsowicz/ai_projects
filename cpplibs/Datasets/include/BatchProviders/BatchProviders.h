#ifndef DATASETS_INCLUDE_BATCHPROVIDERS_BATCHPROVIDERS_H
#define DATASETS_INCLUDE_BATCHPROVIDERS_BATCHPROVIDERS_H

#include <string>
#include <vector>

#include "BatchProviders/IBatchProvider.hpp"

namespace datasets::batchProviders
{
IBatchProviderPtr getFromSerializedPaths(const std::vector<std::string>& paths, bool storeInRam);
} // namespace datasets::batchProviders

#endif
