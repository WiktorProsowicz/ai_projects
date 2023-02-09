#ifndef MLCORE_GRAPHNODES_H
#define MLCORE_GRAPHNODES_H

#include <MLCore/BasicTensor.h>
#include <memory>

namespace mlCore
{
struct Node
{
	Node() = default;
	Node(const TensorPtr tensor)
		: index_(nodesCount_++)
		, value_(tensor){};

	virtual ~Node() = default;

	TensorPtr getValue() const
	{
		return value_;
	}

	uint64_t getIndex() const
	{
		return index_;
	}

	virtual void setValue(const TensorPtr tensor)
	{
		value_ = tensor;
	}
	virtual void setValue(const Tensor& tensor)
	{
		*value_ = tensor;
	}

private:
	uint64_t index_;
	static uint64_t nodesCount_;
	TensorPtr value_;
};

struct Variable : public Node
{
	Variable()
		: Node(){};
	Variable(const TensorPtr tensor)
		: Node(tensor){};
};

struct Constant : public Node
{
	Constant() = delete;
	Constant(const TensorPtr tensor)
		: Node(tensor){};
};

struct Placeholder : public Node
{
	Placeholder()
		: Node(){};
};

} // namespace mlCore
#endif