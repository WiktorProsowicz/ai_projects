#ifndef MLCORE_INCLUDE_MODELS_IOPTIMIZER_HPP
#define MLCORE_INCLUDE_MODELS_IOPTIMIZER_HPP

#include <memory>

#include <AutoDiff/GraphNodes.hpp>

namespace mlCore
{
/**
 * @brief Interface for classes optimizing layers' weights with use of gradients.
 * 
 */
class IOptimizer
{
public:
	/**
     * @brief Modifies the weight with respect to the derivative.
     * 
     * @param weight Weight to be modified.
     * @param derivative Derivative matched to the weight.
     */
	virtual void applyGradient(NodePtr weight, const Tensor& derivative) = 0;

	virtual ~IOptimizer() = default;
};

using IOptimizerPtr = std::shared_ptr<IOptimizer>;
} // namespace mlCore

#endif