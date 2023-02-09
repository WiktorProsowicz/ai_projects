#ifndef MLCORE_BASICTENSOR_H
#define MLCORE_BASICTENSOR_H

#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace mlCore
{
template <typename T>
class BasicTensor
{

public:
	using valueType = T;

	BasicTensor() = delete;
	BasicTensor(const BasicTensor&); // copy constructor
	BasicTensor(BasicTensor&&); // move constructor
	/**
	 * @brief constructs tensor from shape
	 * 
	 */
	BasicTensor(const std::vector<uint64_t>&);
	/**
	 * @brief constructs tensor from shape and fills it with initial value
	 * 
	 */
	BasicTensor(const std::vector<uint64_t>&, const valueType);
	~BasicTensor();

	// assign
	BasicTensor& operator=(const BasicTensor&); // copy assignment
	BasicTensor& operator=(BasicTensor&&); // move assignment
	/**
	 * @brief fills tensor with passed value
	 * 
	 * @return
	 */
	BasicTensor& operator=(const valueType);

	// getters
	const std::vector<uint64_t>& shape() const noexcept
	{
		return shape_;
	}
	uint8_t nDimensions() const noexcept
	{
		return shape_.size();
	}
	size_t size() const noexcept
	{
		return length_;
	}

	// setters
	void reshape(const std::vector<uint64_t>&);
	void assign(std::initializer_list<std::pair<uint64_t, uint64_t>>,
				std::initializer_list<valueType>);

	// operators
	BasicTensor operator+(const BasicTensor&) const;
	BasicTensor operator-(const BasicTensor&) const;
	BasicTensor operator*(const BasicTensor&) const;
	BasicTensor operator/(const BasicTensor&) const;

	BasicTensor matmul(const BasicTensor&) const;

	// displaying
	template <typename TT>
	friend std::ostream& operator<<(std::ostream&, const BasicTensor<TT>&);

private:
	size_t length_;
	std::vector<uint64_t> shape_;
	valueType* data_;
};

template <typename valueType>
std::ostream& operator<<(std::ostream& out, const BasicTensor<valueType>& tensor)
{

	std::function<void(const typename std::vector<valueType>::iterator, const valueType*)>
		recursePrint;
	recursePrint = [&recursePrint, &out, &tensor](const std::vector<valueType>::iterator shapeIter,
												  const valueType* dataPtr) {
		out << "[";
		if(shapeIter == std::prev(tensor.shape_.end()))
		{
			uint64_t i;
			for(i = 0; i < (*shapeIter - 1); i++)
				out << dataPtr[i] << ", ";
			out << dataPtr[i];
		}
		else
		{
			uint64_t i;
			for(i = 0; i < (*shapeIter - 1); i++)
			{
				recursePrint(shapeIter + 1, dataPtr + i * (*shapeIter));
				out << ",\n";
			}
			recursePrint(shapeIter + 1, dataPtr + i * (*shapeIter));
		}
		out << "]";
	};

	recursePrint(tensor.shape_.begin(), tensor.data_);

	return out;
}

using Tensor = BasicTensor<double>;
using TensorPtr = std::shared_ptr<Tensor>;
} // namespace mlCore

#endif