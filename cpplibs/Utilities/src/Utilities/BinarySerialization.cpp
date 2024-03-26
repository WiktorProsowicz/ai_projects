#include "Utilities/BinarySerialization.h"

#include <cstdint>
#include <cstring>

// __Custom preprocessor macros__

/// Template for template makeBytes functions handling c-style strings.
#define REGISTER_MAKEBYTES_CHARS(charType)                                                                   \
	template <>                                                                                              \
	std::string makeBytes(charType object)                                                                   \
	{                                                                                                        \
		return object;                                                                                       \
	}

/// Template for template makeBytes functions handling numeric types.
#define REGISTER_MAKEBYTES_NUMERIC(numType)                                                                  \
	template <>                                                                                              \
	std::string makeBytes(const numType object)                                                              \
	{                                                                                                        \
		std::string stringifiedData(sizeof(object), ' ');                                                    \
		std::memcpy(stringifiedData.data(), &object, sizeof(object));                                        \
		return stringifiedData;                                                                              \
	}

namespace utilities::detail
{

// NOLINTBEGIN
REGISTER_MAKEBYTES_CHARS(const char* const)
REGISTER_MAKEBYTES_CHARS(char* const)
// NOLINTEND

REGISTER_MAKEBYTES_NUMERIC(uint8_t)
REGISTER_MAKEBYTES_NUMERIC(uint16_t)
REGISTER_MAKEBYTES_NUMERIC(uint32_t)
REGISTER_MAKEBYTES_NUMERIC(uint64_t)
REGISTER_MAKEBYTES_NUMERIC(int8_t)
REGISTER_MAKEBYTES_NUMERIC(int16_t)
REGISTER_MAKEBYTES_NUMERIC(int32_t)
REGISTER_MAKEBYTES_NUMERIC(int64_t)
REGISTER_MAKEBYTES_NUMERIC(float)
REGISTER_MAKEBYTES_NUMERIC(double)

template <>
std::string makeBytes(std::string object)
{
	return object;
}
} // namespace utilities::detail
