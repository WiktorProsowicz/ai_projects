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
size_t getNBlocks(std::istream& file)
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
	_fileStream->seekp(0, std::ios_base::beg);

	auto nBlocks = getNBlocks(*_fileStream);
	nBlocks++;

	_fileStream->write(reinterpret_cast<char*>(&nBlocks), sizeof(nBlocks));

	_fileStream->seekp(0, std::ios_base::end);

	(*_fileStream) << utilities::SerializationPack(size_t{tensor.shape().size()});
	(*_fileStream) << utilities::SerializationPack(tensor.shape());
	std::copy(tensor.begin(), tensor.end(), std::ostream_iterator<char>(*_fileStream));

	_tensorHandles.emplace_back(std::make_shared<TensorHandle>(*_fileStream, _fileStream->tellp()));
}

void WeightsSerializer::_validateFile(const std::string& path)
{
	if(std::filesystem::file_size(path) < 8)
	{
		LOG_ERROR("Layers::WeightsSerializer", "File is too small to be a valid weights file.");
	}

	std::ifstream fileStream(path);

	const auto fileEnd = fileStream.seekg(0, std::ios_base::end).tellg();
	std::streampos filePos = fileStream.seekg(8, std::ios_base::beg).tellg();

	size_t nBlocks = getNBlocks(fileStream);

	for(size_t blockIdx = 0; blockIdx < nBlocks; blockIdx++)
	{
		if((fileEnd - filePos) < 8)
		{
			LOG_ERROR("Layers::WeightsSerializer",
					  fmt::format("For block {} could not read the number of dimensions.", blockIdx));
		}

		size_t nDimensions;
		fileStream.readsome(reinterpret_cast<char*>(&nDimensions), sizeof(size_t));
		filePos += 8;
		fileStream.seekg(filePos);

		if(static_cast<size_t>(fileEnd - filePos) < 8 * nDimensions)
		{
			LOG_ERROR("Layers::WeightsSerializer",
					  fmt::format("For block {} not enough data to read dimensions.", blockIdx));
		}

		std::vector<size_t> dimensions(nDimensions);
		fileStream.readsome(reinterpret_cast<char*>(dimensions.data()), 8 * nDimensions);
		filePos += 8 * nDimensions;
		fileStream.seekg(filePos);

		const auto tensorSize =
			std::accumulate(dimensions.begin(), dimensions.end(), size_t{1}, std::multiplies<>());

		if(static_cast<size_t>(fileEnd - filePos) < 8 * tensorSize)
		{
			LOG_ERROR("Layers::WeightsSerializer",
					  fmt::format("For block {} not enough data to read tensor data.", blockIdx));
		}

		filePos += 8 * tensorSize;
		fileStream.seekg(filePos);
	}
}

void WeightsSerializer::_initHandles()
{
	std::streampos currentPos = 8;
	const auto fileEnd = _fileStream->seekg(0, std::ios_base::end).tellg();

	while(currentPos < fileEnd)
	{
		const auto& newestHandle =
			_tensorHandles.emplace_back(std::make_shared<TensorHandle>(*_fileStream, currentPos));

		const auto newestHandleShape = newestHandle->getShape();

		currentPos += 8 + newestHandleShape.size() * 8;
		currentPos += std::accumulate(
			newestHandleShape.cbegin(), newestHandleShape.cend(), size_t{8}, std::multiplies<>());
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

	_file.seekp(_position + static_cast<std::streamoff>(8));
	_file.write(reinterpret_cast<const char*>(tensor.shape().data()), 8 * tensor.shape().size());

	_file.seekp(_position + static_cast<std::streamoff>(8 + 8 * tensor.shape().size()));

	std::copy(tensor.begin(), tensor.end(), std::ostream_iterator<double>(_file));
}

mlCore::Tensor TensorHandle::get() const
{
	const auto shape = getShape();
	mlCore::Tensor tensor(shape);

	_file.seekg(_position + static_cast<std::streamoff>(8 + 8 * shape.size()));

	std::copy_n(std::istream_iterator<double>(_file), tensor.size(), tensor.begin());

	return tensor;
}

mlCore::TensorShape TensorHandle::getShape() const
{
	size_t nDimensions;
	_file.seekg(_position);
	_file.readsome(reinterpret_cast<char*>(&nDimensions), sizeof(size_t));
	_file.seekg(_position + static_cast<std::streamoff>(8));

	std::vector<size_t> dimensions(nDimensions);
	_file.readsome(reinterpret_cast<char*>(dimensions.data()), 8 * nDimensions);

	return dimensions;
}
} // namespace layers::serialization