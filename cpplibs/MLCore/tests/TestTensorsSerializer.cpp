/**********************
 * Test suite for 'ai_projects'
 *
 * Copyright (c) 2024
 *
 * by Wiktor Prosowicz
 **********************/

#include <filesystem>
#include <random>

#include <MLCore/TensorIO/TensorsSerializer.h>
#include <MLCore/TensorOperations.h>
#include <Utilities/BinarySerialization.hpp>
#include <gtest/gtest.h>

namespace
{
/*****************************
 *
 * Common Functions
 *
 *****************************/

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

std::string stringifyShape(const mlCore::TensorShape& shape)
{
	return fmt::format("({})", fmt::join(shape, ", "));
}

::testing::AssertionResult areTensorsEqual(const mlCore::Tensor& expected, const mlCore::Tensor& actual)
{
	if(expected.shape() != actual.shape())
	{
		return ::testing::AssertionFailure() << "Shapes are different: " << stringifyShape(expected.shape())
											 << " vs " << stringifyShape(actual.shape());
	}

	const auto& expectedData = std::vector<double>{expected.begin(), expected.end()};
	const auto& actualData = std::vector<double>{actual.begin(), actual.end()};

	for(size_t i = 0; i < expectedData.size(); ++i)
	{
		if(std::abs(expectedData[i] - actualData[i]) > 1e-6)
		{
			return ::testing::AssertionFailure() << "Tensors are different:\n"
												 << expected << "\n\nvs\n\n"
												 << actual;
		}
	}

	return ::testing::AssertionSuccess();
}

/*****************************
 *
 * Test Fixture
 *
 *****************************/

class TestWeightsSerializer : public ::testing::Test
{
protected:
	void _assertTensorsAsExpected(const std::vector<mlCore::Tensor>& expectedTensors)
	{
		const auto& tensorHandles = _serializer->getTensorHandles();

		ASSERT_EQ(expectedTensors.size(), tensorHandles.size());

		for(size_t i = 0; i < expectedTensors.size(); ++i)
		{
			const auto& tensorHandle = tensorHandles[i];
			const auto& expectedTensor = expectedTensors[i];

			ASSERT_EQ(expectedTensor.shape(), tensorHandle->getShape());

			ASSERT_TRUE(areTensorsEqual(expectedTensor, tensorHandle->get()));
		}
	}

	/// @brief Checks if the path spanned by the serializer contains the expected data.
	template <typename... Args>
	void _assertWeightsFileAsExpected(utilities::SerializationPack<Args...>&& fileData)
	{
		std::ifstream file(_weightsPath, std::ios_base::binary);

		std::ostringstream expectedFileContent(std::ios_base::binary);
		expectedFileContent << fileData;

		std::ostringstream actualFileContent(std::ios_base::binary);
		actualFileContent << file.rdbuf();

		EXPECT_EQ(expectedFileContent.str(), actualFileContent.str());
	}

	/// @brief Checks if the serializer catches an expected error message while validating a file with a given
	/// data.
	template <class... Args>
	::testing::AssertionResult _serializerCatchesInvalidFile(utilities::SerializationPack<Args...>&& fileData,
															 const std::string& expectedErrorMsg)
	{
		const auto tempPath = createTempFile();

		{
			std::ofstream file(tempPath);
			file << fileData;
		}

		try
		{
			mlCore::io::TensorsSerializer::open(tempPath);
		}
		catch(const std::exception& e)
		{
			if(std::string{e.what()}.find(expectedErrorMsg) != std::string::npos)
			{
				return ::testing::AssertionSuccess();
			}

			return ::testing::AssertionFailure() << "Unexpected error message thrown: " << e.what();
		}

		return ::testing::AssertionFailure() << "Expected error message not thrown!";
	}

	void _resetSerializer(const std::string& path)
	{
		_serializer = mlCore::io::TensorsSerializer::open(path);
		_weightsPath = path;
	}

	void _resetSerializer()
	{
		_serializer.reset();
	}

	std::unique_ptr<mlCore::io::TensorsSerializer> _serializer{};
	std::string _weightsPath{};
};

} // namespace

/*****************************
 *
 * Particular test calls
 *
 *****************************/

TEST_F(TestWeightsSerializer, EncountersEmptyFile)
{
	EXPECT_TRUE(_serializerCatchesInvalidFile(utilities::SerializationPack(std::vector<int>{}),
											  "is too small to be a valid weights file"));
}

TEST_F(TestWeightsSerializer, EncountersEmptyBlock)
{
	EXPECT_TRUE(_serializerCatchesInvalidFile(utilities::SerializationPack(uint64_t{1}),
											  "could not read the number of dimensions"));
}

TEST_F(TestWeightsSerializer, EncountersInvalidDimensions)
{
	EXPECT_TRUE(_serializerCatchesInvalidFile(utilities::SerializationPack(uint64_t{1}, size_t{1}),
											  "not enough data to read dimensions"));
}

TEST_F(TestWeightsSerializer, EncountersInvalidData)
{
	EXPECT_TRUE(_serializerCatchesInvalidFile(
		utilities::SerializationPack(uint64_t{1}, size_t{1}, std::vector<size_t>{1}),
		"not enough data to read tensor data"));
}

TEST_F(TestWeightsSerializer, CorrectlyAllocatesTensorsInOneShot)
{
	_resetSerializer(createTempFile());

	mlCore::Tensor tensor1(mlCore::TensorShape{3, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
	mlCore::Tensor tensor2(mlCore::TensorShape{2, 2}, {1.0, 2.0, 3.0, 4.0});

	_serializer->addNewTensor(tensor1);
	_serializer->addNewTensor(tensor2);

	_resetSerializer();

	_assertWeightsFileAsExpected(utilities::SerializationPack{
		uint64_t{2},
		size_t{2},
		std::vector<size_t>{3, 3},
		std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9},
		size_t{2},
		std::vector<size_t>{2, 2},
		std::vector<double>{1, 2, 3, 4},
	});
}

TEST_F(TestWeightsSerializer, CorrectlyAllocatesTensorsInMultipleShots)
{
	const auto weightsPath = createTempFile();

	_resetSerializer(weightsPath);

	mlCore::Tensor tensor1(mlCore::TensorShape{3, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
	mlCore::Tensor tensor2(mlCore::TensorShape{2, 2}, {1.0, 2.0, 3.0, 4.0});

	_serializer->addNewTensor(tensor1);

	_resetSerializer(weightsPath);

	_serializer->addNewTensor(tensor2);

	_resetSerializer();

	_assertWeightsFileAsExpected(utilities::SerializationPack{
		uint64_t{2},
		size_t{2},
		std::vector<size_t>{3, 3},
		std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9},
		size_t{2},
		std::vector<size_t>{2, 2},
		std::vector<double>{1, 2, 3, 4},
	});
}

TEST_F(TestWeightsSerializer, CorrectlyDecodesTensors)
{
	const auto weightsPath = createTempFile();

	_resetSerializer(weightsPath);

	const mlCore::Tensor tensor1(mlCore::TensorShape{3, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
	const mlCore::Tensor tensor2(mlCore::TensorShape{2, 2}, {1.0, 2.0, 3.0, 4.0});

	_serializer->addNewTensor(tensor1);
	_serializer->addNewTensor(tensor2);

	_resetSerializer(weightsPath);

	_assertTensorsAsExpected({tensor1, tensor2});
}

TEST_F(TestWeightsSerializer, CorrectlyUpdatesTensors)
{
	const auto weightsPath = createTempFile();

	_resetSerializer(weightsPath);

	const mlCore::Tensor tensor1(mlCore::TensorShape{3, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
	const mlCore::Tensor tensor2(mlCore::TensorShape{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
	const mlCore::Tensor tensor3(mlCore::TensorShape{3, 2}, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0});

	_serializer->addNewTensor(tensor1);
	_serializer->addNewTensor(tensor2);

	_resetSerializer(weightsPath);

	_serializer->getTensorHandles()[1]->save(tensor3);

	_assertTensorsAsExpected({tensor1, tensor3});
}