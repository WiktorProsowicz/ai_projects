#ifndef AUTODIFF_GRAPHOPERATIONS_H
#define AUTODIFF_GRAPHOPERATIONS_H

#include <AutoDiff/GraphNodes.h>

namespace mlCore
{
struct BinaryOperations
{
	NodePtr multiply(const NodePtr lNode, const NodePtr rNode) const;
	NodePtr add(const NodePtr lNode, const NodePtr rNode) const;
	NodePtr subtract(const NodePtr lNode, const NodePtr rNode) const;
	NodePtr divide(const NodePtr lNode, const NodePtr rNode) const;
	NodePtr matmul(const NodePtr lNode, const NodePtr rNode) const;
};

struct UnaryOperations
{
	NodePtr power(const NodePtr baseNode, const NodePtr factorNode) const;
};

struct NodesActivations
{
	NodePtr relu(const NodePtr node) const;
	NodePtr sigmoid(const NodePtr node) const;
};

struct DerivativeExtractor
{
	const Tensor operator()(const UnaryOperPtr oper) const;
	const std::pair<Tensor, Tensor> operator()(const BinaryOperPtr oper) const;
};
} // namespace mlCore

#endif