#include "LayersModules/Activations.h"

#include <AutoDiff/Operations.h>

#include "LayersModules/SimpleActivation.hpp"

namespace layers::activations
{
IActivationFactoryPtr relu()
{
	const auto reluFunc = [](const autoDiff::NodePtr& input) -> autoDiff::OperatorPtr
	{ return autoDiff::ops::relu(input); };

	return std::make_unique<detail::SimpleActivation>(reluFunc, "ReLU");
}

IActivationFactoryPtr sigmoid()
{
	const auto sigmoidFunc = [](const autoDiff::NodePtr& input) -> autoDiff::OperatorPtr
	{ return autoDiff::ops::sigmoid(input); };

	return std::make_unique<detail::SimpleActivation>(sigmoidFunc, "Sigmoid");
}
} // namespace layers::activations