#ifndef MLCORE_INCLUDE_MODELS_ILAYER_HPP
#define MLCORE_INCLUDE_MODELS_ILAYER_HPP

#include <memory>

#include "AutoDiff/GraphNodes.hpp"
#include "MLCore/BasicTensor.h"

namespace interfaces
{
/**
 * @brief Interface for models' components. Defines a scope inside a model rather than a linear layer inside
 * NN.
 *
 */
class ILayer
{
public:
	ILayer() = default;

	ILayer(const ILayer&) = default;
	ILayer(ILayer&&) = default;
	ILayer& operator=(const ILayer&) = default;
	ILayer& operator=(ILayer&&) = default;

	virtual ~ILayer() = default;

	/**
	 * @brief Calls the layer and returns the output value.
	 *
	 * @details The output value should be computed based on the input values and the characteristics concrete
	 * layer's implementation. The internal value of the output operator shall be updated before returning it.
	 *
	 * @param inputs Input values.
	 */
	virtual autoDiff::OperatorPtr call(const std::vector<autoDiff::NodePtr>& inputs) = 0;

	/**
	 * @brief Returns the shape of the layer's output.
	 *
	 */
	virtual mlCore::TensorShape getOutputShape() const = 0;

	/**
	 * @brief Gives the layer's weights that are supposed to be trained.
	 *
	 * @return Layer's trainable weights.
	 */
	virtual std::vector<autoDiff::NodePtr> getTrainableWeights() const = 0;

	/**
	 * @brief Compiles textual description of the layer based on its type and parameters.
	 *
	 * @return Textual description of the layer.
	 */
	virtual std::string getDescription() const = 0;
};

using ILayerPtr = std::shared_ptr<ILayer>;
} // namespace interfaces

#endif
