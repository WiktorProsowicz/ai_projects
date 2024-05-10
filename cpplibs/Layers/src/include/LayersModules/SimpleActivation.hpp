#ifndef LAYERS_SRC_INCLUDE_LAYERSMODULES_SIMPLEACTIVATION_HPP
#define LAYERS_SRC_INCLUDE_LAYERSMODULES_SIMPLEACTIVATION_HPP

#include "LayersModules/IActivationFactory.hpp"

namespace layers::detail
{
using ActivationFunction = std::function<autoDiff::OperatorPtr(const autoDiff::NodePtr&)>;

/**
 * @brief Applies a given function to the input node.
 *
 */
class SimpleActivation : public IActivationFactory
{
public:
	SimpleActivation() = delete;

	SimpleActivation(const ActivationFunction& activationFunc, const std::string& description)
		: _description(description)
		, _activationFunc(activationFunc)
	{}

	SimpleActivation(const SimpleActivation&) = default;
	SimpleActivation(SimpleActivation&&) = default;
	SimpleActivation& operator=(const SimpleActivation&) = default;
	SimpleActivation& operator=(SimpleActivation&&) = default;

	~SimpleActivation() override = default;

	autoDiff::OperatorPtr apply(const autoDiff::NodePtr& input) override
	{
		return _activationFunc(input);
	}

	const std::string& getDescription() const override
	{
		return _description;
	}

private:
	std::string _description;
	std::function<autoDiff::OperatorPtr(const autoDiff::NodePtr&)> _activationFunc;
};
} // namespace layers::detail

#endif