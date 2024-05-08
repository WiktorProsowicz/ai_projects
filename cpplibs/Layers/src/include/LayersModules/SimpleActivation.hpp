#ifndef LAYERS_SRC_INCLUDE_LAYERSMODULES_SIMPLEACTIVATION_HPP
#define LAYERS_SRC_INCLUDE_LAYERSMODULES_SIMPLEACTIVATION_HPP

#include "LayersModules/IActivationFactory.hpp"

namespace layers::detail
{
/**
 * @brief Applies a given function to the input node.
 *
 */
class SimpleActivation : public IActivationFactory
{
public:
	SimpleActivation() = delete;

	SimpleActivation(const SimpleActivation&) = default;
	SimpleActivation(SimpleActivation&&) = default;
	SimpleActivation& operator=(const SimpleActivation&) = default;
	SimpleActivation& operator=(SimpleActivation&&) = default;

	~SimpleActivation() override = default;

	autoDiff::OperatorPtr apply(const autoDiff::NodePtr& input) override;

	const std::string& getDescription() const override;

private:
	std::function<>
};
} // namespace layers::detail

#endif