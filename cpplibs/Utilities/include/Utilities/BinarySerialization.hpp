#ifndef UTILITIES_INCLUDE_UTILITIES_BINARYSERIALIZATION_H
#define UTILITIES_INCLUDE_UTILITIES_BINARYSERIALIZATION_H

#include <any>
#include <list>
#include <ostream>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace utilities
{
namespace detail
{
template <typename T>
concept UnhandledSerializable = requires(T object) {
	requires !std::is_same_v<const char*, T>;
	requires !std::is_same_v<char*, T>;
	requires !std::ranges::range<T>;
};

template <typename Type>
std::string makeBytes(Type* object)
	requires std::disjunction_v<std::is_same<Type, const char>, std::is_same<Type, char>>;

template <std::ranges::range Type>
std::string makeBytes(const Type& object);

/**
 * @brief Template function used to define custom conversion-to-bytes mechanics for chosen types.
 * Purpose of the function is to provide means of representing objects as byte-arrays containing data
 * relevant to the reader of the serialized version, rather than the direct underlying bytes.
 *
 * @tparam Type Type of the object to be converted into byte representation.
 * @param object Object to be serialized.
 * @return Short-char string containing bytes extracted from the `object`.
 */
template <UnhandledSerializable Type>
std::string makeBytes(Type object);

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
	SerializationPack(SerializationPack&&) = delete;				 // Move constructor.
	SerializationPack& operator=(const SerializationPack&) = delete; // Copy assignment.
	SerializationPack& operator=(SerializationPack&&) = delete;		 // Move assignment.

	~SerializationPack() = default;

	/**
	 * @brief Constructs the SerializationPack from given arguments and initializes the internal objects pack,
	 * adjusting types if needed.
	 *
	 * @param args Object to pack.
	 */
	explicit SerializationPack(const ArgTypes&... args)
		: _args(std::list<std::any, std::allocator<std::any>>{std::decay_t<ArgTypes>(args)...})
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
		const auto& frontArg = *(std::next(_args.cbegin(), static_cast<int64_t>(_poppedArgIndex)));

		_poppedArgIndex++;

		return std::any_cast<std::decay_t<CastedType>>(frontArg);
	}

	/**
	 * @brief Writes contained elements in form of extracted bytes to the given stream.
	 *
	 * @param out Stream to write the elements to.
	 */
	void _packToStream(std::ostream& out) const
	{
		_poppedArgIndex = 0;

		(out << ... << detail::makeBytes(_popSingleArg<ArgTypes>()));
	}

private:
	std::list<std::any> _args;
	mutable size_t _poppedArgIndex = 0;
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

namespace detail
{

template <typename Type>
std::string makeBytes(Type* const object)
	requires std::disjunction_v<std::is_same<Type, const char>, std::is_same<Type, char>>
{
	return object;
}

template <std::ranges::range Type>
std::string makeBytes(const Type& object)
{
	std::ostringstream strComposer;

	for(const auto& item : object)
	{
		strComposer << makeBytes(item);
	}

	return strComposer.str();
}

template <UnhandledSerializable Type>
std::string makeBytes(Type object)
{
	const std::span<uint8_t> underlyingMemory(reinterpret_cast<uint8_t*>(&object), sizeof(object));

	return {underlyingMemory.begin(), underlyingMemory.end()};
}
} // namespace detail

} // namespace utilities

#endif
