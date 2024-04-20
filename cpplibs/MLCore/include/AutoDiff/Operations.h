#ifndef MLCORE_INCLUDE_OPERATIONS_H
#define MLCORE_INCLUDE_OPERATIONS_H

#include "AutoDiff/GraphNodes.hpp"

namespace autoDiff::ops
{
/**
 * @brief Performs the addition operation on the two nodes.
 */
OperatorPtr add(const NodePtr& lhsNode, const NodePtr& rhsNode);

/**
 * @brief Performs the subtraction operation on the two nodes.
 */
OperatorPtr subtract(const NodePtr& lhsNode, const NodePtr& rhsNode);

/**
 * @brief Performs the multiplication operation on the two nodes.
 */
OperatorPtr multiply(const NodePtr& lhsNode, const NodePtr& rhsNode);

/**
 * @brief Performs the division operation on the two nodes.
 */
OperatorPtr divide(const NodePtr& lhsNode, const NodePtr& rhsNode);

/**
 * @brief Performs matrix multiplication on the input nodes.
 */
OperatorPtr matmul(const NodePtr& lhsNode, const NodePtr& rhsNode);

/**
 * @brief Applies natural logarithm function on the input node.
 */
OperatorPtr naturalLog(const NodePtr& node);

/**
 * @brief Applies Rectified Linear Unit (ReLU) function on the input node.
 *
 * The RELU function returns the input value if it is positive, otherwise it returns 0.
 */
OperatorPtr relu(const NodePtr& node);

/**
 * @brief Applies Sigmoid function on the input node.
 *
 * The sigmoid function squeezes the input values between 0 and 1.
 */
OperatorPtr sigmoid(const NodePtr& node);
} // namespace autoDiff::ops

#endif
