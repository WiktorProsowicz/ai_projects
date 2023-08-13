#ifndef MLCORE_INCLUDE_MODELS_ILAYER_HPP
#define MLCORE_INCLUDE_MODELS_ILAYER_HPP

#include <memory>

#include <AutoDiff/GraphNodes.hpp>
#include <MLCore/BasicTensor.h>

namespace mlCore::models
{
/**
 * @brief Interface for models' components. Defines a scope inside a model rather than a linear layer inside NN.
 * 
 */
class ILayer
{
public:
	/**
     * @brief Initializes the layer's parameters based on the specifically defined configuration. 
     * 
     * @return Expected layer's output.
     */
	virtual autoDiff::NodePtr build() = 0;

	/**
      * @brief Computes the layer's output.
      * 
      * @return Computed output. 
      */
	virtual Tensor compute() = 0;

	/**
      * @brief Gives the layer's weights regardless of whether they are trainable or not.
      * 
      * @return Layer's weights.
      */
	virtual std::vector<autoDiff::NodePtr> getAllWeights() const = 0;

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

	virtual ~ILayer() = default;
};

using ILayerPtr = std::shared_ptr<ILayer>;
} // namespace mlCore::models

#endif