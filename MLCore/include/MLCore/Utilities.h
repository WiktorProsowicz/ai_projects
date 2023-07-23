#ifndef MLCORE_UTILITIES_H
#define MLCORE_UTILITIES_H

#include <iterator>
#include <limits>
#include <sstream>
#include <vector>

namespace mlCore
{

template <typename T>
std::string stringifyVector(const std::vector<T>& vect,
							const char* const openSign = "(",
							const char* const closeSign = ")")
{
	std::ostringstream serializingStream;
	serializingStream << openSign;
	std::copy(vect.cbegin(), vect.cend(), std::ostream_iterator<T>(serializingStream, ", "));
	serializingStream << closeSign;
	return serializingStream.str();
}

} // namespace mlCore

#endif