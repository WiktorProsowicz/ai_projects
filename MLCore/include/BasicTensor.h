#ifndef MLCORE_BASICTENSOR_H
#define MLCORE_BASICTENSOR_H

namespace mlCore
{
template <typename T>
class BasicTensor
{
public:
	BasicTensor() = delete;
	BasicTensor(const BasicTensor&);
	BasicTensor(BasicTensor&&);
	~BasicTensor() = default;
	BasicTensor& operator=(const BasicTensor&);
	BasicTensor& operator=(BasicTensor&&);
};

using Tensor = BasicTensor<double>;
} // namespace mlCore

#endif