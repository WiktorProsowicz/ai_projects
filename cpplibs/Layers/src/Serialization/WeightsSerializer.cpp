#include "Serialization/WeightsSerializer.h"

#include <filesystem>

#include <Utilities/BinarySerialization.hpp>

namespace layers::serialization
{
std::unique_ptr<WeightsSerializer> WeightsSerializer::open(const std::string& path)
{
	if(!std::filesystem::exists(path))
	{
		std::ofstream file(path);
		file << utilities::SerializationPack(size_t{0});
	}
	else
	{
		_validateFile(path);
	}

	std::unique_ptr<std::fstream> fileStream = std::make_unique<std::fstream>(
		path, std::ios_base::app | std::ios_base::in | std::ios_base::out | std::ios_base::binary);

	return std::make_unique<WeightsSerializer>(std::move(fileStream));
}

void WeightsSerializer::_validateFile(const std::string& path)
{
	if(std::filesystem::file_size(path) < 8)
	{
		LOG_ERROR("Layers::WeightsSerializer", "File is too small to be a valid weights file.");
	}

	std::ifstream fileStream(path);
	std::streampos filePos = fileStream.seekg(8, std::ios_base::beg).tellg();

	size_t nBlocks;
	fileStream.readsome(reinterpret_cast<char*>(&nBlocks), sizeof(size_t));

	for(size_t blockIdx = 0; blockIdx < nBlocks; blockIdx++)
	{
        
	}
}
} // namespace layers::serialization