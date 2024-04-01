/**********************
 * Test suite for 'ai_projects'
 *
 * Copyright (c) 2023
 *
 * by Wiktor Prosowicz
 **********************/

#include <fstream>
#include <sstream>

#include <Utilities/BinarySerialization.hpp>
#include <fmt/format.h>
#include <gtest/gtest.h>

/*****************************
 *
 * Common functions
 *
 *****************************/

namespace
{
/**
 * @brief Compares serialized bytes in form of char-string with vector of expected bytes.
 *
 * @param str Serialized bytes.
 * @param bytes Expected bytes.
 */
void compareStringifiedOutputWithBytes(const std::string& str, const std::vector<uint8_t>& bytes)
{

	ASSERT_EQ(str.size(), bytes.size())
		<< "Can't compare stringified input with expected bytes in case they have different sizes.";

	auto strIt = str.cbegin();
	auto bytesIt = bytes.cbegin();

	for(; (strIt < str.cend()) && (bytesIt < bytes.cend()); strIt++, bytesIt++)
	{
		ASSERT_EQ(static_cast<uint8_t>(*strIt), *bytesIt);
	}
}
} // namespace

/*****************************
 *
 * Particular test calls
 *
 *****************************/

/**
 * @brief Streams an example serialization pack and compares the output with expected bytes.
 *
 */
TEST(TestSerializationPack, testSerializationWithExpectedOutput)
{
	const utilities::SerializationPack pack(
		"abcde",
		uint8_t{123},
		uint16_t{12345},
		uint32_t{123456789},
		uint64_t{1234567890},
		int8_t{-123},
		int16_t{-12345},
		int32_t{-123456789},
		int64_t{-123456789},
		float{123.456},
		double{123456.7890},
		std::string{"fghj"},
		std::vector<uint16_t>{123, 124, 125, 126},
		std::vector<std::string>{"abc", "efg"},
		std::vector<std::vector<std::string>>{{"a", "b", "c"}, {"d", "e", "f"}});

	const std::vector<uint8_t> expectedBytes{
		0x61, 0x62, 0x63, 0x64, 0x65, 0x7b, 0x39, 0x30, 0x15, 0xcd, 0x5b, 0x07, 0xd2, 0x02, 0x96,
		0x49, 0x00, 0x00, 0x00, 0x00, 0x85, 0xc7, 0xcf, 0xeb, 0x32, 0xa4, 0xf8, 0xeb, 0x32, 0xa4,
		0xf8, 0xff, 0xff, 0xff, 0xff, 0x79, 0xe9, 0xf6, 0x42, 0xc9, 0x76, 0xbe, 0x9f, 0x0c, 0x24,
		0xfe, 0x40, 0x66, 0x67, 0x68, 0x6a, 0x7b, 0x00, 0x7c, 0x00, 0x7d, 0x00, 0x7e, 0x00, 0x61,
		0x62, 0x63, 0x65, 0x66, 0x67, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66};

	std::stringstream serializationStream;

	serializationStream << pack;

	utilities::detail::makeBytes(std::vector<std::string>{"aaa", "ccc"});

	compareStringifiedOutputWithBytes(serializationStream.str(), expectedBytes);
}
