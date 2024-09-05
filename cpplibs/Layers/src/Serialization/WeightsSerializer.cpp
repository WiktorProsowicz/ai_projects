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

	std::unique_ptr<std::fstream> fileStream =
		std::make_unique<std::fstream>(path, std::ios_base::in | std::ios_base::out | std::ios_base::binary);

	return std::unique_ptr<WeightsSerializer>{new WeightsSerializer{std::move(fileStream)}};
}

WeightsSerializer::WeightsSerializer(std::unique_ptr<std::fstream> fileStream)
	: _fileStream(std::move(fileStream))
{
	_initHandles();
}

namespace
{
/// Returns the number of tensors stored inside the analyzed file.
uint64_t getNBlocks(std::istream& file)
{
	const auto initPos = file.tellg();
	size_t nBlocks;

	file.seekg(0, std::ios_base::beg);
	file.read(reinterpret_cast<char*>(&nBlocks), sizeof(nBlocks));
	file.seekg(initPos);

	return nBlocks;
}
} // namespace

void WeightsSerializer::addNewTensor(const mlCore::Tensor& tensor)
{
	uint64_t nBlocks = getNBlocks(*_fileStream);
	nBlocks++;

	_fileStream->seekp(0, std::ios_base::beg);
	(*_fileStream) << utilities::SerializationPack(nBlocks);

	_fileStream->seekp(0, std::ios_base::end);
	(*_fileStream) << utilities::SerializationPack(size_t{tensor.shape().size()});
	(*_fileStream) << utilities::SerializationPack(tensor.shape());

	for(const auto value : tensor)
	{
		_fileStream->write(reinterpret_cast<const char*>(&value), sizeof(double));
	}

	_fileStream->flush();
	_tensorHandles.emplace_back(std::make_shared<TensorHandle>(*_fileStream, _fileStream->tellp()));
}

void WeightsSerializer::_validateFile(const std::string& path)
{
	if(std::filesystem::file_size(path) < sizeof(uint64_t))
	{
		LOG_ERROR("Layers::WeightsSerializer",
				  fmt::format("File '{}' is too small to be a valid weights file.", path));
	}

	std::ifstream fileStream(path);

	const auto fileEnd = fileStream.seekg(0, std::ios_base::end).tellg();
	std::streampos filePos = fileStream.seekg(sizeof(uint64_t), std::ios_base::beg).tellg();

	size_t nBlocks = getNBlocks(fileStream);

	for(size_t blockIdx = 0; blockIdx < nBlocks; blockIdx++)
	{
		if(static_cast<size_t>(fileEnd - filePos) < sizeof(size_t))
		{
			LOG_ERROR("Layers::WeightsSerializer",
					  fmt::format("For block {} in file '{}' could not read the number of dimensions.",
								  blockIdx,
								  path));
		}

		size_t nDimensions;
		fileStream.readsome(reinterpret_cast<char*>(&nDimensions), sizeof(size_t));
		filePos += sizeof(uint64_t);
		fileStream.seekg(filePos);

		if(static_cast<size_t>(fileEnd - filePos) < (sizeof(size_t) * nDimensions))
		{
			LOG_ERROR(
				"Layers::WeightsSerializer",
				fmt::format("For block {} in file '{}' not enough data to read dimensions.", blockIdx, path));
		}

		std::vector<size_t> dimensions(nDimensions);
		fileStream.readsome(reinterpret_cast<char*>(dimensions.data()), sizeof(size_t) * nDimensions);
		filePos += sizeof(size_t) * nDimensions;
		fileStream.seekg(filePos);

		const auto tensorSize =
			std::accumulate(dimensions.begin(), dimensions.end(), size_t{1}, std::multiplies<>());

		if(static_cast<size_t>(fileEnd - filePos) < sizeof(double) * tensorSize)
		{
			LOG_ERROR("Layers::WeightsSerializer",
					  fmt::format(
						  "For block {} in file '{}' not enough data to read tensor data.", blockIdx, path));
		}

		filePos += sizeof(double) * tensorSize;
		fileStream.seekg(filePos);
	}
}

void WeightsSerializer::_initHandles()
{
	std::streampos currentPos = sizeof(uint64_t);

	for(size_t blockIdx = 0; blockIdx < getNBlocks(*_fileStream); blockIdx++)
	{
		const auto& newestHandle =
			_tensorHandles.emplace_back(std::make_shared<TensorHandle>(*_fileStream, currentPos));

		const auto newestHandleShape = newestHandle->getShape();

		currentPos += sizeof(size_t) + newestHandleShape.size() * sizeof(size_t);
		currentPos += std::accumulate(newestHandleShape.cbegin(),
									  newestHandleShape.cend(),
									  size_t{sizeof(double)},
									  std::multiplies<>());
	}
}

TensorHandle::TensorHandle(std::fstream& file, std::streampos position)
	: _file(file)
	, _position(position)
{}

void TensorHandle::save(const mlCore::Tensor& tensor)
{
	const auto currentShape = getShape();

	const auto allowedTensorSize =
		std::accumulate(currentShape.cbegin(), currentShape.cend(), size_t{1}, std::multiplies<>());

	if(tensor.size() != allowedTensorSize)
	{
		LOG_ERROR("Layers::WeightsSerializer",
				  fmt::format("Tensor size does not match the expected size. Expected: {}, got: {}.",
							  allowedTensorSize,
							  tensor.size()));
	}

	_file.seekp(_position + static_cast<std::streamoff>(sizeof(size_t)));
	_file << utilities::SerializationPack(tensor.shape());

	_file.seekp(_position +
				static_cast<std::streamoff>(sizeof(size_t) + sizeof(size_t) * tensor.shape().size()));

	for(const auto value : tensor)
	{
		_file.write(reinterpret_cast<const char*>(&value), sizeof(double));
	}

	_file.flush();
}

mlCore::Tensor TensorHandle::get() const
{
	const auto shape = getShape();
	mlCore::Tensor tensor(shape);

	_file.seekg(_position + static_cast<std::streamoff>(sizeof(size_t) + sizeof(size_t) * shape.size()));

	for(auto& value : tensor)
	{
		_file.read(reinterpret_cast<char*>(&value), sizeof(double));
	}

	return tensor;
}

mlCore::TensorShape TensorHandle::getShape() const
{
	size_t nDimensions;
	_file.seekg(_position);
	_file.readsome(reinterpret_cast<char*>(&nDimensions), sizeof(size_t));
	_file.seekg(_position + static_cast<std::streamoff>(sizeof(size_t)));

	std::vector<size_t> dimensions(nDimensions);
	_file.readsome(reinterpret_cast<char*>(dimensions.data()), sizeof(size_t) * nDimensions);

	return dimensions;
}
} // namespace layers::serialization