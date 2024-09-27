#ifndef LAYERS_TESTS_UTILITIES_HPP
#define LAYERS_TESTS_UTILITIES_HPP

#include <filesystem>
#include <random>

#include <LoggingLib/LoggingLib.hpp>
#include <MLCore/BasicTensor.h>
#include <MLCore/TensorIO/TensorsSerializer.h>
#include <fmt/format.h>
#include <gtest/gtest.h>

namespace testUtilities
{
/// @brief Creates a temporary file and returns its path.
std::string createTempFile()
{
	static constexpr size_t maxAttempts = 10;

	std::random_device randomDev;
	std::uniform_int_distribution<uint64_t> randomDist(0, std::numeric_limits<uint64_t>::max());
	std::mt19937 randomEngine(randomDev());

	for(uint8_t attempt = 0; attempt < maxAttempts; ++attempt)
	{

		const auto tempPath =
			std::filesystem::temp_directory_path() / std::to_string(randomDist(randomEngine));

		if(!std::filesystem::exists(tempPath))
		{
			return tempPath;
		}
	}

	LOG_ERROR("TestUtilities", "Critical! Failed to create a temporary file!");
}

/// @brief Checks if two tensors are equal.
::testing::AssertionResult areTensorsEqual(const mlCore::Tensor& tensor1, const mlCore::Tensor& tensor2)
{
	if(tensor1.shape() != tensor2.shape())
	{
		return ::testing::AssertionFailure() << "Shapes of the tensors are different!";
	}

	if(!std::equal(tensor1.begin(), tensor1.end(), tensor2.begin()))
	{
		return ::testing::AssertionFailure() << "Tensors are different:\n"
											 << tensor1 << "\n\nvs\n\n"
											 << tensor2;
	}

	return ::testing::AssertionSuccess();
}

/// @brief Checks if a given file exists and contains the expected content.
::testing::AssertionResult areSavedWeightsAsExpected(const std::string& weightsPath,
													 const std::vector<mlCore::Tensor>& expectedTensors)
{
	if(!std::filesystem::exists(weightsPath))
	{
		return ::testing::AssertionFailure() << fmt::format("File '{}' does not exist!", weightsPath);
	}

	const auto weightsFileHandle = mlCore::io::TensorsSerializer::open(weightsPath);
	const auto tensorHandles = weightsFileHandle->getTensorHandles();

	if(tensorHandles.size() != expectedTensors.size())
	{
		return ::testing::AssertionFailure()
			   << fmt::format("Number of tensors in the file ({}) does not match the expected number ({})!",
							  tensorHandles.size(),
							  expectedTensors.size());
	}

	for(size_t i = 0; i < tensorHandles.size(); ++i)
	{
		const auto& savedTensor = tensorHandles[i]->get();
		const auto& expectedTensor = expectedTensors[i];

		if(savedTensor.shape() != expectedTensor.shape())
		{
			return ::testing::AssertionFailure()
				   << fmt::format("Shape of tensor {} does not match the expected shape!", i);
		}

		if(!std::equal(savedTensor.begin(), savedTensor.end(), expectedTensor.begin()))
		{
			return ::testing::AssertionFailure() << "Tensors are different:\n"
												 << expectedTensor << "\n\nvs\n\n"
												 << savedTensor;
		}
	}

	return ::testing::AssertionSuccess();
}
} // namespace testUtilities

#endif