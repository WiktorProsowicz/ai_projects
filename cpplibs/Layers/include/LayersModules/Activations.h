#ifndef LAYERS_INCLUDE_LAYERSMODULES_ACTIVATIONS_H
#define LAYERS_INCLUDE_LAYERSMODULES_ACTIVATIONS_H

#include "LayersModules/IActivationFactory.hpp"

namespace layers::activations
{
/**
 * @brief Returns an object that applies the ReLU function to the input node.
 */
IActivationFactoryPtr relu();

/**
 * @brief Returns an object that applies the sigmoid function to the input node.
 */
IActivationFactoryPtr sigmoid();

} // namespace layers::activations

#endif