#ifndef DATASETS_INCLUDE_BATCHPROVIDERS_BATCHPROVIDERS_H
#define DATASETS_INCLUDE_BATCHPROVIDERS_BATCHPROVIDERS_H

#include "BatchProviders/IBatchProvider.hpp"

namespace datasets::batchProviders
{
IBatchProviderPtr getFromSerializedPaths(const std::vector<std::string>& paths, bool storeInRam);
}

#endif