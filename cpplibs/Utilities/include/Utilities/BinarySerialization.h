#ifndef UTILITIES_INCLUDE_UTILITIES_BINARYSERIALIZATION_H
#define UTILITIES_INCLUDE_UTILITIES_BINARYSERIALIZATION_H

#include <any>
#include <list>
#include <ostream>
#include <string>
#include <vector>

namespace utilities
{
namespace detail
{
/**
 * @brief Template function used to define custom conversion-to-bytes mechanics for chosen types.
 * Purpose of the function is to provide means of representing objects as byte-arrays containing data
 * relevant to the reader of the serialized version, rather than the direct underlying bytes.
 *
 * @tparam Type Type of the object to be converted into byte representation.
 * @param object Object to be serialized.
 * @return Short-char string containing bytes extracted from the `object`.
 */
template <typename Type>
std::string makeBytes(Type object);

template <typename VectorItemType>
std::string makeBytes(const std::vector<VectorItemType>& object)
{
	std::string ss;

	for(const auto& item : object)
	{
		ss += makeBytes(item);
	}

	return ss;
}

} // namespace detail

/**
 * @brief Class sued to wrap objects of various types. It defines means of conversion of the objects to the
 * pre-defined binary forms.
 *
 */
template <typename... ArgTypes>
class SerializationPack
{
public:
	SerializationPack() = delete;									 // Default constructor.
	SerializationPack(const SerializationPack&) = delete;			 // Copy constructor.
	SerializationPack(SerializationPack&) = delete;					 // Move constructor.
	SerializationPack& operator=(const SerializationPack&) = delete; // Copy assignment.
	SerializationPack& operator=(SerializationPack&&) = delete;		 // Move assignment.

	/**
	 * @brief Constructs the SerializationPack from given arguments and initializes the internal objects pack,
	 * adjusting types if needed.
	 *
	 * @param args Object to pack.
	 */
	SerializationPack(const ArgTypes&... args)
		: args_(std::list<std::any, std::allocator<std::any>>{std::decay_t<ArgTypes>(args)...})
	{}

	template <typename... PackArgTypes>
	friend std::ostream& operator<<(std::ostream& out, const SerializationPack<PackArgTypes...>& pack);

private:
	/**
	 * @brief Attempts to retrieve and cast an element from internal objects pack.
	 * The number of returned element is indicated by the current value of `poppedArgIndex_`.
	 *
	 * @tparam CastedType Desired type to cast the retrieved element to.
	 * @return Casted element.
	 */
	template <typename CastedType>
	std::decay_t<CastedType> _popSingleArg() const
	{
		auto& frontArg = *(std::next(args_.cbegin(), static_cast<int64_t>(poppedArgIndex_)));

		poppedArgIndex_++;

		return std::any_cast<std::decay_t<CastedType>>(frontArg);
	}

	/**
	 * @brief Writes contained elements in form of extracted bytes to the given stream.
	 *
	 * @param out Stream to write the elements to.
	 */
	void _packToStream(std::ostream& out) const
	{
		poppedArgIndex_ = 0;

		(out << ... << detail::makeBytes(_popSingleArg<ArgTypes>()));
	}

private:
	std::list<std::any> args_;
	mutable size_t poppedArgIndex_ = 0;
};

/**
 * @brief Writes the contents of the `pack` to the given stream. Pack's elements are converted
 * to the internal binary representation according to the custom converting algorithms.
 *
 * @param out Stream to write the `pack`'s content to.
 * @param pack SerializationPack containing elements to convert and write.
 * @return std::ostream&
 */
template <typename... ArgTypes>
std::ostream& operator<<(std::ostream& out, const SerializationPack<ArgTypes...>& pack)
{
	pack._packToStream(out);

	return out;
}

} // namespace utilities

#endif
