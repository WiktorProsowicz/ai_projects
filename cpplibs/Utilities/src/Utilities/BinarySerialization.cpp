// __Related headers__
#include <Utilities/BinarySerialization.h>

// __C++ standard headers__
#include <cstring>

// __Custom preprocessor macros__

/// Template for template makeBytes functions handling c-style strings.
#define __REGISTER_MAKEBYTES_CHARS(charType)                                                                                     \
	template <>                                                                                                                  \
	std::string makeBytes(charType object)                                                                                       \
	{                                                                                                                            \
		return object;                                                                                                           \
	}

/// Template for template makeBytes functions handling numeric types.
#define __REGISTER_MAKEBYTES_NUMERIC(numType)                                                                                    \
	template <>                                                                                                                  \
	std::string makeBytes(const numType object)                                                                                  \
	{                                                                                                                            \
		std::string s(sizeof(object), ' ');                                                                                      \
		std::memcpy(s.data(), &object, sizeof(object));                                                                          \
		return s;                                                                                                                \
	}

namespace utilities
{
namespace detail
{
__REGISTER_MAKEBYTES_CHARS(const char*)
__REGISTER_MAKEBYTES_CHARS(char*)

__REGISTER_MAKEBYTES_NUMERIC(uint8_t)
__REGISTER_MAKEBYTES_NUMERIC(uint16_t)
__REGISTER_MAKEBYTES_NUMERIC(uint32_t)
__REGISTER_MAKEBYTES_NUMERIC(uint64_t)
__REGISTER_MAKEBYTES_NUMERIC(int8_t)
__REGISTER_MAKEBYTES_NUMERIC(int16_t)
__REGISTER_MAKEBYTES_NUMERIC(int32_t)
__REGISTER_MAKEBYTES_NUMERIC(int64_t)
__REGISTER_MAKEBYTES_NUMERIC(float)
__REGISTER_MAKEBYTES_NUMERIC(double)

template <>
std::string makeBytes(std::string object)
{
	return object;
}
} // namespace detail
} // namespace utilities
