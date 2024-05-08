#ifndef LAYERS_INCLUDE_LAYERSMODULES_IACTIVATIONFACTORY_HPP
#define LAYERS_INCLUDE_LAYERSMODULES_IACTIVATIONFACTORY_HPP

#include <AutoDiff/GraphNodes.hpp>

namespace layers
{
/**
 * @brief Provides an interface for classes applying activation functions to graph nodes.
 *
 * @details Activation functions are simply operators connected to nodes being the output of a layer. The
 * classes implementing IActivationFactory function shall apply the activation according to internal
 * parameters and concrete activation type.
 */
class IActivationFactory
{
public:
	IActivationFactory() = default;

	IActivationFactory(const IActivationFactory&) = default;
	IActivationFactory(IActivationFactory&&) = default;
	IActivationFactory& operator=(const IActivationFactory&) = default;
	IActivationFactory& operator=(IActivationFactory&&) = default;

	virtual ~IActivationFactory() = default;

	/**
	 * @brief Applies the activation function to the input node.
	 *
	 * @param input Node to apply the activation to.
	 * @return Node with the activation applied. The internal output node's value shall be updated.
	 */
	virtual autoDiff::OperatorPtr apply(const autoDiff::NodePtr& input) = 0;

	/**
	 * @brief Returns the string identifier of the activation function.
	 *
	 */
	virtual const std::string& getDescription() const = 0;
};

using IActivationFactoryPtr = std::unique_ptr<IActivationFactory>;
} // namespace layers

#endif