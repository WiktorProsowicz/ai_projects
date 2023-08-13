#ifndef MLCORE_UTILITIES_H
#define MLCORE_UTILITIES_H

#include <iterator>
#include <limits>
#include <sstream>
#include <vector>

#include <fmt/format.h>

namespace mlCore
{

template <typename T>
std::string stringifyVector(const std::vector<T>& vect, const char* const openSign = "(", const char* const closeSign = ")")
{
	return fmt::format("{}{}{}", openSign, fmt::join(vect, ", "), closeSign);
}

} // namespace mlCore

#endif